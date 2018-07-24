# API How to: https://github.com/IntelRealSense/librealsense/wiki/API-How-To
import json
import os
import time
import traceback

import numpy as np
import pyrealsense2 as rs

from cone_detection.cone_detection import generate_canny, get_cones


def start_pipeline(advanced_mode=False, fps=30, width=640, height=480, preset_file=None, record_to_file=None,
                   from_file=None):
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        if record_to_file is not None:
            config.enable_record_to_file(record_to_file)
        elif from_file is not None:
            config.enable_device_from_file(from_file)
        profile = pipeline.start(config)
        device = profile.get_device()  # type: rs.device

        # Note: Depth and Color sensor options are shared, so setting them on one sets them on both
        # depth_sensor = device.first_depth_sensor()  # type: rs.depth_sensor
        # set_high_density_mode(depth_sensor)
        # depth_sensor.set_option(rs.option.exposure, 66000)
        # depth_sensor.set_option(rs.option.frames_queue_size, 2)
    except:
        print('exception in start_pipeline(), before starting pipeline')
        traceback.print_exc()
        return False

    try:
        print('Started pipeline at depth fps: %s, color fps: %s' % get_fps_values(profile))
        if advanced_mode:
            dev = find_device_that_supports_advanced_mode()
            if dev is not None:
                enable_advanced_mode(dev, preset_file)
        return pipeline
    except:
        print('exception in start_pipeline()')
        traceback.print_exc()
        stop_pipeline(pipeline)
        return False


def print_option_data(profile):
    device = profile.get_device()  # type: rs.device
    depth_sensor = device.first_depth_sensor()  # type: rs.depth_sensor
    color_sensor = device.first_roi_sensor()  # type: rs.roi_sensor
    all_options = [rs.option.accuracy,
                   rs.option.asic_temperature,
                   rs.option.auto_exposure_converge_step,
                   rs.option.auto_exposure_mode,
                   rs.option.auto_exposure_priority,
                   rs.option.backlight_compensation,
                   rs.option.brightness,
                   rs.option.color_scheme,
                   rs.option.confidence_threshold,
                   rs.option.contrast,
                   rs.option.depth_units,
                   rs.option.emitter_enabled,
                   rs.option.enable_auto_exposure,
                   rs.option.enable_auto_white_balance,
                   rs.option.enable_motion_correction,
                   rs.option.error_polling_enabled,
                   rs.option.exposure,
                   rs.option.filter_magnitude,
                   rs.option.filter_option,
                   rs.option.filter_smooth_alpha,
                   rs.option.filter_smooth_delta,
                   rs.option.frames_queue_size,
                   rs.option.gain,
                   rs.option.gamma,
                   rs.option.histogram_equalization_enabled,
                   rs.option.holes_fill,
                   rs.option.hue,
                   rs.option.laser_power,
                   rs.option.max_distance,
                   rs.option.min_distance,
                   rs.option.motion_module_temperature,
                   rs.option.motion_range,
                   rs.option.output_trigger_enabled,
                   rs.option.power_line_frequency,
                   rs.option.projector_temperature,
                   rs.option.saturation,
                   rs.option.sharpness,
                   rs.option.stereo_baseline,
                   rs.option.texture_source,
                   rs.option.total_frame_drops,
                   rs.option.visual_preset,
                   rs.option.white_balance]

    def print_options(options, sensor, title):
        print("***************%s*******************************" % title)
        for opt in options:
            try:
                print('%s (%s)' % (opt, sensor.get_option_description(opt)))
            except RuntimeError as e:
                continue
            _range = sensor.get_option_range(opt)  # type: rs.option_range
            print('\tDefault  :%s (%s)' % (_range.default, sensor.get_option_value_description(opt, _range.default)))
            current_value = sensor.get_option(opt)
            print('\tCurrent  :%s (%s)' % (current_value, sensor.get_option_value_description(opt, current_value)))
            if _range.max == _range.min or _range.step == 0 or (_range.max - _range.min) / _range.step > 10:
                print('\tMIN value:%s' % _range.min)
                print('\tMAX value:%s' % _range.max)
                print('\tSTEP valu:%s' % _range.step)
            else:
                for i in np.arange(_range.min, _range.max + _range.step, _range.step):
                    opt_desc = sensor.get_option_value_description(opt, i)
                    print('\t%s.\t\t%s' % (i, opt_desc))
            print('\n')

    print_options(all_options, depth_sensor, "Depth Sensor Options")
    print_options(all_options, color_sensor, "Color Sensor Options")


def get_fps_values(profile):
    dstream = profile.get_stream(rs.stream.depth)  # type:rs.stream_profile
    cstream = profile.get_stream(rs.stream.color)  # type:rs.stream_profile
    return dstream.fps(), cstream.fps()


def get_depth_sensor(pipeline):
    profile = pipeline.get_active_profile()  # type:rs.pipeline_profile
    device = profile.get_device()  # type:rs.device
    depth_sensor = device.first_depth_sensor()  # type:rs.depth_sensor
    return depth_sensor


def get_fov(pipeline):
    profile = pipeline.get_active_profile()  # type:rs.pipeline_profile
    stream = profile.get_stream(rs.stream.depth)  # type:rs.stream_profile
    video_stream = stream.as_video_stream_profile()  # type: rs.video_stream_profile
    intrinsics = video_stream.get_intrinsics()  # type: rs.intrinsics
    fov_list = rs.rs2_fov(intrinsics);
    return fov_list


def get_depth_scale(pipeline):
    return get_depth_sensor(pipeline).get_depth_scale()


def print_advanced_mode_settings(adv_mode):
    json_str = adv_mode.serialize_json()
    parsed = json.loads(json_str)
    print(json.dumps(parsed, indent=4, sort_keys=True))


def get_advanced_mode_preset_json(preset_file):
    with open(os.path.join('', preset_file)) as f:
        data = json.load(f)
        data = json.dumps(data)
    return data


def enable_advanced_mode(dev, preset_file=None):
    adv_mode = rs.rs400_advanced_mode(dev)
    print("Advanced mode is", "enabled" if adv_mode.is_enabled() else "disabled")
    while not adv_mode.is_enabled():
        print("Trying to enable advanced mode...")
        adv_mode.toggle_advanced_mode(True)
        # At this point the device will disconnect and re-connect.
        print("Sleeping for 5 seconds...")
        time.sleep(5)
        # The 'dev' object will become invalid and we need to initialize it again
        dev = find_device_that_supports_advanced_mode()
        adv_mode = rs.rs400_advanced_mode(dev)
        print("Advanced mode is", "enabled" if adv_mode.is_enabled() else "disabled")

    if preset_file is not None:
        config = get_advanced_mode_preset_json(preset_file)
        adv_mode.load_json(config)

    print_advanced_mode_settings(adv_mode)


### Safely stop. If the rs_pipeline is null, it is started first, then stopped.

def stop_pipeline(rs_pipeline):
    if rs_pipeline is None:
        rs_pipeline = rs.pipeline()
    rs_pipeline.stop()


def read_n_frames(pipeline, n):
    for i in range(n):
        pipeline.wait_for_frames()


def get_raw_frames(pipeline, align_to=None):
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    if align_to is not None:
        frames = align_to.process(frames)

    depth_frame = frames.get_depth_frame()  # type:rs.depth_frame
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        raise Exception()
    return depth_frame, color_frame


def get_frames_from_raw(depth_frame, color_frame):
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    return depth_image, color_image


def get_frames(pipeline, align_to=None):
    depth_frame, color_frame = get_raw_frames(pipeline, align_to)
    return get_frames_from_raw(depth_frame, color_frame)


def find_device_that_supports_advanced_mode():
    ds5_product_ids = ["0AD1", "0AD2", "0AD3", "0AD4", "0AD5", "0AF6", "0AFE", "0AFF", "0B00", "0B01", "0B03", "0B07"]
    ctx = rs.context()
    devices = ctx.query_devices()
    for dev in devices:
        if dev.supports(rs.camera_info.product_id) and str(dev.get_info(rs.camera_info.product_id)) in ds5_product_ids:
            if dev.supports(rs.camera_info.name):
                print("Found device that supports advanced mode:", dev.get_info(rs.camera_info.name))
            return dev
    return None


# Deprecated and should be unused. Use the preset json file instead.
def set_high_density_mode(sensor):
    _range = sensor.get_option_range(rs.option.visual_preset)  # type: rs.option_range
    for i in np.arange(_range.min, _range.max, _range.step):
        if sensor.get_option_value_description(rs.option.visual_preset, i) == "High Density":
            sensor.set_option(rs.option.visual_preset, i)


def detect_once(color_image):
    try:
        rs_canny = generate_canny(color_image)
        list_of_cones, _unused = get_cones(rs_canny)
        return list_of_cones
    except:
        traceback.print_exc()


# Constants for holes_fill
FILL_ALL_ZERO_PIXELS = 5


def get_depth_filter_list(decimate=True, d2d=True, spatial=True, temporal=True):
    filters = []
    if decimate:
        dec_filt = rs.decimation_filter()
        dec_filt.set_option(rs.option.filter_magnitude, 2)
        filters.append(dec_filt)

    if d2d:
        depth2disparity = rs.disparity_transform()
        filters.append(depth2disparity)

    if spatial:
        spat = rs.spatial_filter()
        spat.set_option(rs.option.holes_fill, FILL_ALL_ZERO_PIXELS)
        filters.append(spat)

    if temporal:
        temp = rs.temporal_filter()
        filters.append(temp)

    if d2d:
        disparity2depth = rs.disparity_transform(False)
        filters.append(disparity2depth)

    return filters


def apply_filters(filters, frame):
    for rs_filter in filters:
        frame = rs_filter.process(frame)
    return frame
