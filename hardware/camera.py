import logging
import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs
logger = logging.getLogger(__name__)
class RealSenseCamera:
    def __init__(self,
                 device_id=None,
                 width=640,
                 height=480,
                 fps=30):  # Changed from 6 to 30 FPS (supported by D405)
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.scale = None
        self.intrinsics = None
    def get_available_devices(self):
        """Get list of available RealSense devices"""
        ctx = rs.context()
        devices = ctx.query_devices()
        available_devices = []
        for device in devices:
            serial = device.get_info(rs.camera_info.serial_number)
            name = device.get_info(rs.camera_info.name)
            available_devices.append({'serial': serial, 'name': name})
        return available_devices
    def connect(self):
        # Check for available devices if no specific device_id provided
        available_devices = self.get_available_devices()
        if not available_devices:
            raise RuntimeError("No RealSense devices found")
        if self.device_id is None:
            # Use the first available device
            self.device_id = available_devices[0]['serial']
            logger.info(f"No device ID specified, using first available device: {self.device_id} ({available_devices[0]['name']})")
        else:
            # Check if specified device is available
            device_serials = [dev['serial'] for dev in available_devices]
            if str(self.device_id) not in device_serials:
                logger.warning(f"Requested device {self.device_id} not found. Available devices: {device_serials}")
                self.device_id = available_devices[0]['serial']
                logger.info(f"Using first available device: {self.device_id} ({available_devices[0]['name']})")
        # Start and configure
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(str(self.device_id))
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        try:
            cfg = self.pipeline.start(config)
            logger.info(f"Successfully connected to RealSense device: {self.device_id}")
        except Exception as e:
            logger.error(f"Failed to start pipeline: {e}")
            raise RuntimeError(f"Could not connect to camera {self.device_id}: {e}")
        # Determine intrinsics
        rgb_profile = cfg.get_stream(rs.stream.color)
        self.intrinsics = rgb_profile.as_video_stream_profile().get_intrinsics()
        # Determine depth scale
        self.scale = cfg.get_device().first_depth_sensor().get_depth_scale()
    def get_image_bundle(self):
        frames = self.pipeline.wait_for_frames()
        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.first(rs.stream.color)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        depth_image = np.asarray(aligned_depth_frame.get_data(), dtype=np.float32)
        depth_image *= self.scale
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.expand_dims(depth_image, axis=2)
        return {
            'rgb': color_image,
            'aligned_depth': depth_image,
        }
    def plot_image_bundle(self):
        images = self.get_image_bundle()
        rgb = images['rgb']
        depth = images['aligned_depth']
        fig, ax = plt.subplots(1, 2, squeeze=False)
        ax[0, 0].imshow(rgb)
        m, s = np.nanmean(depth), np.nanstd(depth)
        ax[0, 1].imshow(depth.squeeze(axis=2), vmin=m - s, vmax=m + s, cmap=plt.cm.gray)
        ax[0, 0].set_title('rgb')
        ax[0, 1].set_title('aligned_depth')
        plt.show()
if __name__ == '__main__':
    cam = RealSenseCamera()
    cam.connect()
    while True:
        cam.plot_image_bundle()
