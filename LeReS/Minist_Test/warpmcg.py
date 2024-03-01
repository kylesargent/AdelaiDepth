import math
import os
from glob import glob
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from einops import rearrange, repeat
from PIL import Image
from torchvision.transforms import ToTensor

import torch
import numpy as np
from PIL import Image
import torch
import pytorch3d
import matplotlib.pyplot as plt
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    BlendParams,
    HardPhongShader,
    look_at_view_transform,
    SoftSilhouetteShader,
)
# from pytorch3d.renderer import SfMPerspectiveCameras as PerspectiveCameras
from pytorch3d.renderer import PerspectiveCameras as PerspectiveCameras
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from pytorch3d.renderer.mesh.textures import TexturesVertex
# from pytorch3d.renderer.mesh.textures import TexturesVertex
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import warpinp

matmul = np.matmul
mat_vec_mul = lambda A, b: matmul(A, b[..., None])[..., 0]

from torch import nn


class AlbedoShader(nn.Module):
    def __init__(self, device="cpu", n_color_channels=3):
        super().__init__()
        self.device = device
        self.n_color_channels = n_color_channels

    def forward(self, fragments, meshes, **kwargs):
        texels = meshes.sample_textures(fragments)
        valid_mask = (fragments.pix_to_face >= 0).float()
        images = texels * valid_mask.unsqueeze(-1)
        images = images[..., 0, :]

        alpha_mask = images[..., self.n_color_channels :]
        images = images[..., : self.n_color_channels]
        valid_mask *= alpha_mask

        return torch.concat([images, valid_mask], axis=-1)


def align_depth(colmap_depth, estimated_dense_disparity):
    assert colmap_depth.shape == estimated_dense_disparity.shape, (
        colmap_depth.shape,
        estimated_dense_disparity.shape,
    )
    disparity_max = 1000
    disparity_min = 0.001
    depth_max = 1 / disparity_min
    depth_min = 1 / disparity_max
    mask = colmap_depth != 0

    target = colmap_depth
    target = torch.clip(target, depth_min, depth_max)

    prediction = estimated_dense_disparity

    target_disparity = torch.zeros_like(target)
    target_disparity[mask == 1] = 1.0 / target[mask == 1]

    scale, shift = compute_scale_and_shift(prediction, target_disparity, mask)

    prediction_aligned = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)
    # prediction_aligned = (prediction_aligned - gt_shift.view(-1, 1, 1)) / (
    #     gt_scale.view(-1, 1, 1)
    # )

    prediction_aligned[prediction_aligned > disparity_max] = disparity_max
    prediction_aligned[prediction_aligned < disparity_min] = disparity_min
    prediction_depth = 1.0 / prediction_aligned
    return prediction_depth, scale, shift


def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def rerender_rgbd(
    image,
    depth,
    intrinsics,
    main_extrinsics,
    novel_extrinsics,
    downsample_f=1,
    face_prune_threshold=None,
):
    h, w, c = image.shape
    h1, w1, one = depth.shape

    assert h == h1, w == w1
    assert one == 1
    assert main_extrinsics.shape == (4, 4)
    assert novel_extrinsics.shape[1:] == (4, 4)

    intrinsics = np.linalg.inv(intrinsics)
    intrinsics[:2] /= downsample_f
    intrinsics = np.linalg.inv(intrinsics)

    image = cv2.resize(
        image, (w // downsample_f, h // downsample_f), interpolation=cv2.INTER_AREA
    )
    depth = cv2.resize(
        depth, (w // downsample_f, h // downsample_f), interpolation=cv2.INTER_NEAREST
    )[..., None]

    mesh_np = get_mesh(
        image,
        depth,
        intrinsics,
        main_extrinsics,
        face_prune_threshold=face_prune_threshold,
    )

    vertices = torch.from_numpy(mesh_np["vertices"]).to(torch.float32).cuda()
    faces = torch.from_numpy(mesh_np["triangles"]).cuda()
    textures = torch.from_numpy(mesh_np["textures"]).cuda().to(torch.float32)
    textures = TexturesVertex(verts_features=textures[None])

    mesh = Meshes(verts=[vertices], faces=[faces], textures=textures)
    camtopix = np.linalg.inv(intrinsics)
    if w >= h:
        focal_length = camtopix[1, 1] / camtopix[1, 2]
    else:
        raise NotImplementedError

    raster_settings = RasterizationSettings(
        image_size=(h // downsample_f, w // downsample_f),
        # blur_radius=0,
        # bin_size=None,
        # perspective_correct=True,
        # max_faces_per_bin=1_000_000,
    )

    rasterizer = MeshRasterizer(raster_settings=raster_settings)
    renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=AlbedoShader(device="cuda", n_color_channels=c),
    )
    focal_length = float(focal_length)
    # print(focal_length)

    renders = []
    for novel_extrinsic in novel_extrinsics:
        worldtocam = np.linalg.inv(novel_extrinsic)
        worldtocam[0] *= -1  # opengl -> pytorch3d
        worldtocam[2] *= -1
        R = (worldtocam[:3, :3].T)[None]
        T = worldtocam[:3, -1][None]
        T += 1e-6  # add some small noise because the renderer is quite buggy

        cameras = PerspectiveCameras(device="cuda", R=R, T=T, focal_length=focal_length)

        render = renderer(mesh, cameras=cameras)
        renders.append(render)
    return renders


def pixel_coords(w, h):
    return np.meshgrid(np.arange(w), np.arange(h), indexing="xy")


def fi(t, i=0):
    if t.ndim == 4:
        t = t[i]
    t = np.array(t)  # .transpose((1,2,0))
    t = np.clip(t * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(t)


def get_homogeneous_point_cloud(image, depth, intrinsics, extrinsics):
    h, w, c = image.shape
    # assert h == w
    assert depth.shape == (h, w, 1)
    assert intrinsics.shape == (4, 4)
    assert extrinsics.shape == (4, 4)

    n = h * w

    ## first, image, depth -> point cloud
    pixel_x, pixel_y = pixel_coords(w, h)
    pixelspace_locations = np.stack(
        [pixel_x + 0.5, pixel_y + 0.5, np.ones_like(pixel_x)], axis=-1
    )
    pixelspace_locations = pixelspace_locations.reshape((-1, 3))
    color = image.reshape((-1, c))
    assert color.shape == (n, c)

    pixtocam = intrinsics[:3, :3]
    camtoworld = extrinsics

    cameraspace_locations = mat_vec_mul(pixtocam, pixelspace_locations)
    cameraspace_locations = matmul(
        cameraspace_locations, np.diag(np.array([1.0, -1.0, -1.0]))
    )
    cameraspace_locations *= depth.reshape((-1, 1))

    homogeneous_cameraspace_locations = np.concatenate(
        [cameraspace_locations, np.ones_like(cameraspace_locations[..., :1])],
        axis=-1,
    )

    homogeneous_worldspace_locations = mat_vec_mul(
        camtoworld, homogeneous_cameraspace_locations
    )

    return homogeneous_worldspace_locations, color


def get_normals(depth, intrinsics, extrinsics):
    focal = np.linalg.inv(intrinsics)[0, 0]
    # focal length in pixels

    assert depth.ndim == 3
    assert depth.shape[2] == 1
    depth = np.clip(depth[..., 0], a_min=1e-3, a_max=None)

    dz_dv, dz_du = np.gradient(depth)
    # u*depth = fx*x + cx --> du/dx = fx / depth
    du_dx = focal / depth  # x is xyz of camera coordinate
    dv_dy = focal / depth

    dz_dx = dz_du * du_dx
    dz_dy = dz_dv * dv_dy
    # cross-product (1,0,dz_dx)X(0,1,dz_dy) = (-dz_dx, -dz_dy, 1)
    normal = np.dstack((-dz_dx, -dz_dy, np.ones_like(depth)))
    n = np.linalg.norm(normal, axis=2, keepdims=True)
    # normal /= jnp.clip(n, 1e-3)
    normal /= n

    normal = normal.reshape((-1, 3, 1))
    # normal = normal[..., 0]
    transform = np.matmul(
        extrinsics[:3, :3],
        np.diag(np.array([-1, 1, 1])),
    )

    # todo vectorize it
    corrected_normals = []
    for uncorrected_normal in normal:
        corrected_normal = np.matmul(transform, uncorrected_normal)
        corrected_normals.append(corrected_normal)
    corrected_normals = np.array(corrected_normals)[..., 0]
    return corrected_normals


def get_mesh(image, depth, intrinsics, extrinsics, face_prune_threshold=None):
    h, w, c = image.shape
    # assert h == w

    normals = get_normals(depth, intrinsics, extrinsics)

    homogeneous_points, colors = get_homogeneous_point_cloud(
        image, depth, intrinsics, extrinsics
    )

    if face_prune_threshold is None:
        alphas = np.ones_like(colors[..., :1])
    else:
        viewdir = extrinsics[:3, 2]
        alphas = (
            np.abs(np.matmul(viewdir, normals[..., None]))[..., 0]
            > face_prune_threshold
        )[..., None]

    vertices = homogeneous_points[..., :3]

    def get_triangles(i, j):
        # don't mess with this as it's important for winding order
        unraveled_face_coordinates = np.array(
            [
                [[i, j], [i + 1, j + 1], [i + 1, j]],
                [[i, j], [i, j + 1], [i + 1, j + 1]],
            ]
        )
        raveled_face_coordinates = (
            unraveled_face_coordinates[..., 1] * w + unraveled_face_coordinates[..., 0]
        )

        assert raveled_face_coordinates.shape == (
            2,
            3,
        ), raveled_face_coordinates.shape
        return raveled_face_coordinates

    ys, xs = np.meshgrid(np.arange(h - 1), np.arange(w - 1))
    # triangles = jax.vmap(get_triangles)(xs.ravel(), ys.ravel())
    triangles = np.array(
        [get_triangles(x, y) for (x, y) in zip(xs.ravel(), ys.ravel())]
    )
    triangles = triangles.reshape((-1, 3))
    # print(alphas.shape, image.shape)
    return dict(
        vertices=vertices,
        triangles=triangles,
        textures=np.concatenate(
            [image.reshape((-1, c)), alphas.reshape((-1, 1))], axis=-1
        ),
    )


def load_depthanything():
    image_processor = AutoImageProcessor.from_pretrained(
        "LiheYoung/depth-anything-large-hf"
    )
    model = AutoModelForDepthEstimation.from_pretrained(
        "LiheYoung/depth-anything-large-hf"
    )

    # prepare image for the model
    return model, image_processor


def run_depthanything(image, model, image_processor):
    inputs = image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # visualize the prediction
    output = prediction.squeeze().cpu().numpy()
    return output * 255 / np.max(output)


def central_crop_img_arr(img):
    h, w, c = img.shape
    # assert min(h, w) == 256
    s = min(h,w)
    oh_resid = (h - s) % 2
    ow_resid = (w - s) % 2
    oh = (h - s) // 2
    ow = (w - s) // 2
    img = img[oh : h - oh - oh_resid, ow : w - ow - ow_resid]
    # assert img.shape == (256, 256, c), img.shape
    return img


def run_and_correct_depth(image_pil, state):
    pred_ssi_disparity_depthanything = run_depthanything(
        image_pil, state['depthanything_model'], state['depthanything_processor'],
    )

    pred_si_depth_leres, focal_length, _ = warpinp.process_image(*state['leres_models'], np.array(image_pil))
    # return pred_ssi_disparity_depthanything, pred_si_depth_leres, focal_length

    pred_si_depth_depthanything, scale, shift = align_depth(
        torch.from_numpy(pred_si_depth_leres)[None],
        torch.from_numpy(pred_ssi_disparity_depthanything)[None],
    )
    pred_si_depth_depthanything = pred_si_depth_depthanything[0]

    return pred_si_depth_depthanything, focal_length


def compute_intrinsics(focal_length, h):
    camtopix = np.array([
        [focal_length, 0, h//2, 0],
        [0, focal_length, h//2, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return np.linalg.inv(camtopix)