import cv2
import numpy as np
import open3d as o3d
import rospy

try:
    import sophuspy as sp
except Exception as e:
    print(f"Cannot import sophuspy due to {e}. Import sophus instead")
    import sophus as sp

from spot_wrapper.spot import Spot, SpotCamIds, image_response_to_cv2

intel_img_src = [SpotCamIds.INTEL_REALSENSE_COLOR, SpotCamIds.INTEL_REALSENSE_DEPTH]  # type: ignore
gripper_img_src = [SpotCamIds.HAND_COLOR, SpotCamIds.HAND_DEPTH_IN_HAND_COLOR_FRAME]


def get_intel_image(spot: Spot):
    return spot.get_image_responses(intel_img_src, quality=100, await_the_resp=True)  # type: ignore


def get_gripper_image(spot: Spot):
    return spot.get_image_responses(gripper_img_src, quality=100, await_the_resp=True)  # type: ignore


# Function to capture points
def capture_correspondences(img1, img2):
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
            text = f"{len(points)}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            position = (x, y)  # Position of the text (bottom-left corner)
            font_scale = 1
            font_color = (255, 255, 255)  # White color
            thickness = 2
            line_type = cv2.LINE_AA

            # Write the text on the image
            cv2.putText(
                param,
                text,
                position,
                font,
                font_scale,
                font_color,
                thickness,
                line_type,
            )

            cv2.imshow("Image", param)

    cv2.imshow("Gripper Image", img1)
    cv2.imshow("Intel Image", img2)
    cv2.setMouseCallback("Gripper Image", click_event, img1)
    cv2.setMouseCallback("Intel Image", click_event, img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return points


def sample_patch_around_point(
    cx: int, cy: int, depth_raw: np.ndarray, patch_size: int = 5
) -> int:
    """
    Samples a median depth in 5x5 patch around given x, y (pixel location in depth image array) as center in raw depth image
    """
    h, w = depth_raw.shape
    x1, x2 = cx - patch_size // 2, cx + patch_size // 2
    y1, y2 = cy - patch_size // 2, cy + patch_size // 2
    x1, x2 = np.clip([x1, x2], 0, w)
    y1, y2 = np.clip([y1, y2], 0, h)
    depth_raw = np.nan_to_num(depth_raw)
    deph_patch = depth_raw[y1:y2, x1:x2]
    deph_patch = deph_patch[deph_patch > 0.0]
    return np.median(deph_patch)


def get_3d_points(depth, points, intrinsics):
    fx, fy, cx, cy = intrinsics
    points_3d = []
    for u, v in points:
        Z = sample_patch_around_point(int(u), int(v), depth)
        assert Z > 0.0
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        points_3d.append([X, Y, Z])
    return np.array(points_3d)


def create_open3d_rgbd(rgb_image, depth_image):
    # Ensure the RGB image is in the correct shape and type
    if (
        rgb_image.dtype != np.uint8
        or len(rgb_image.shape) != 3
        or rgb_image.shape[2] != 3
    ):
        raise ValueError("RGB image must be of shape HxWx3 and dtype uint8")

    # Ensure the depth image is in the correct shape and type
    # if depth_image.dtype != np.uint16 or len(depth_image.shape) != 2:
    # raise ValueError("Depth image must be of shape HxW and dtype uint16")

    # Convert the RGB image to an Open3D image
    rgb_o3d = o3d.geometry.Image(rgb_image)

    # Convert the depth image to an Open3D image
    depth_o3d = o3d.geometry.Image(depth_image.astype(np.float32))

    # Create an Open3D RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=rgb_o3d, depth=depth_o3d, convert_rgb_to_intensity=False
    )

    return rgbd_image


def create_pinhole_camera_intrinsic(fx, fy, cx, cy, width, height):
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)
    return intrinsic


if __name__ == "__main__":
    spot: Spot = Spot("Calibration")

    intel_response = get_intel_image(spot)
    # intel_response = intel_response.result()
    intel_intrinsics = intel_response[0].source.pinhole.intrinsics
    intel_intrinsics = (
        intel_intrinsics.focal_length.x,
        intel_intrinsics.focal_length.y,
        intel_intrinsics.principal_point.x,
        intel_intrinsics.principal_point.y,
    )
    intel_image: np.ndarray = image_response_to_cv2(intel_response[0])
    intel_depth: np.ndarray = (
        image_response_to_cv2(intel_response[1]) / 1000.0
    ).astype(np.float32)
    height, width = intel_depth.shape

    gripper_response = get_gripper_image(spot)
    # gripper_response = gripper_response.result()
    gripper_intrinsics = gripper_response[0].source.pinhole.intrinsics
    gripper_intrinsics = (
        gripper_intrinsics.focal_length.x,
        gripper_intrinsics.focal_length.y,
        gripper_intrinsics.principal_point.x,
        gripper_intrinsics.principal_point.y,
    )
    gripper_image: np.ndarray = image_response_to_cv2(gripper_response[0])
    gripper_depth: np.ndarray = (
        image_response_to_cv2(gripper_response[1]) / 1000.0
    ).astype(np.float32)

    # Capture user-defined correspondences
    print(
        "Select corresponding points in the Gripper image and then in the Intel image"
    )
    correspondences = capture_correspondences(gripper_image, intel_image)

    N = len(correspondences) // 2
    gripper_points = correspondences[:N]
    intel_points = correspondences[N:]

    # Get 3D points
    intel_points_3d = get_3d_points(intel_depth, intel_points, intel_intrinsics)
    breakpoint()
    gripper_points_3d = get_3d_points(gripper_depth, gripper_points, gripper_intrinsics)

    gripper_pcd = o3d.geometry.PointCloud()
    gripper_pcd.points = o3d.utility.Vector3dVector(gripper_points_3d)

    intel_pcd = o3d.geometry.PointCloud()
    intel_pcd.points = o3d.utility.Vector3dVector(intel_points_3d)

    correspondence = np.asarray([[i, i] for i in range(N)])
    transformation = o3d.pipelines.registration.TransformationEstimationPointToPoint().compute_transformation(
        gripper_pcd, intel_pcd, o3d.utility.Vector2iVector(correspondence)
    )

    print("Initial Transformation Matrix:\n", transformation)
    gripper_pcd_full = o3d.geometry.PointCloud.create_from_rgbd_image(
        create_open3d_rgbd(gripper_image, gripper_depth),
        create_pinhole_camera_intrinsic(*gripper_intrinsics, width, height),
    )
    intel_pcd_full = o3d.geometry.PointCloud.create_from_rgbd_image(
        create_open3d_rgbd(intel_image, intel_depth),
        create_pinhole_camera_intrinsic(*intel_intrinsics, width, height),
    )

    # Apply ICP
    icp_result = o3d.pipelines.registration.registration_icp(
        intel_pcd_full,
        gripper_pcd_full,
        max_correspondence_distance=0.05,
        init=transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )

    print("Refined Transformation Matrix:\n", icp_result.transformation)
