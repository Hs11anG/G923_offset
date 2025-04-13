import sys
sys.path.append('../logidrivepy')
from logidrivepy import LogitechController
import time
import random
import ctypes
import pandas as pd
from datetime import datetime

def random_turn_controller(controller):
    # 初始化參數
    last_trigger_time = time.time()
    straight_start_time = None
    straight_threshold = 30.0  # 直行角度閾值
    correction_threshold = 5.0  # 校正時回到 pre_angle 的正負範圍
    offset_threshold = 6.0  # 偏移角度需超過的閾值
    straight_duration_min = 5.0  # 直行時間最小值
    straight_duration_max = 6.0  # 直行時間最大值
    force_duration = 0.3  # 力的持續時間
    applying_force = False
    force_start_time = 0
    target_offset = 0 
    change_degree = 120  # 拉的角度
    damp_force = 50  # 阻尼力道
    force_magnitude = 100  # 彈簧力力道
    force_slope = 80  # 力回饋斜率
    throttle_threshold = 0.9  # 油門閾值
    throttle_delay = 1.0  # 油門踩下後的延遲時間
    last_throttle_press_time = 0  # 上次油門踩下的時間
    
    # Excel 記錄初始化
    log_data = []
    excel_filename = f"steering_log_{datetime.now().strftime('%Y%m%d_%H_%M')}.xlsx"

    #保證方向盤運作正常
    for i in range(-50, 0, 2):
        controller.LogiPlaySpringForce(0, i, 100, 40)
        controller.logi_update()
        time.sleep(0.1)
    controller.steering_initialize()
    controller.LogiStopSpringForce(0)
    controller.logi_update()
    # 檢查方向盤旋轉範圍
    range_val = ctypes.c_int()
    if controller.get_operating_range(0, range_val):
        steering_range = range_val.value
        print(f"檢測到方向盤旋轉範圍: {steering_range} 度")
    else:
        steering_range = 900
        print("無法獲取方向盤旋轉範圍，假設為 900 度")

    # 啟動阻尼力
    controller.LogiPlayDamperForce(0, damp_force)
    controller.logi_update()
    print("已啟用阻尼力，提供持續阻力。")

    print("\n---Logitech 隨機轉向測試 (80 度)---")
    print(f"當油門踩下 (< {throttle_threshold}) 並直行 5-10 秒 (± {straight_threshold} 度) 後，方向盤將隨機轉向 80 度。")

    try:
        while True:
            controller.logi_update()
            current_time = time.time()

            if applying_force:
                if current_time - force_start_time >= force_duration:  # 力回饋時間到
                    controller.LogiStopSpringForce(0)
                    controller.LogiPlayDamperForce(0, damp_force)
                    controller.logi_update()
                    applying_force = False

                    # 記錄偏移結束時的角度
                    state = controller.get_state_engines(0).contents
                    current_lx = state.lX
                    current_angle = (current_lx / 32768) * (steering_range / 2)
                    start_angle = current_angle
                    print(f"彈簧力停止，起始角度: {start_angle:.2f} 度")

                    # 計算偏移角度
                    offset_angle = start_angle - pre_angle
                    fail_offset = "V" if abs(offset_angle) <= offset_threshold else ""
                    if fail_offset:
                        print(f"偏移角度過小 ({abs(offset_angle):.2f} 度 <= {offset_threshold} 度)，標記 FAIL_OFFSET 為 'V'")

                    # 開始追蹤校正時間和角度
                    correction_start_time = current_time
                    angles_at_intervals = []
                    fine_angles = []
                    interval_times = [0.1, 0.2, 0.3, 0.4, 0.5]
                    fine_interval_times = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
                    recorded_intervals = []
                    fine_recorded_intervals = []
                    reaction_time = None
                    max_correction_time = 3.0  # 最大校正時間3秒

                    while reaction_time is None:
                        controller.logi_update()
                        state = controller.get_state_engines(0).contents
                        current_lx = state.lX
                        current_angle = (current_lx / 32768) * (steering_range / 2)
                        elapsed_time = time.time() - correction_start_time

                        # 記錄 0.1 到 0.5 秒的角度
                        for t in interval_times:
                            if t not in recorded_intervals and elapsed_time >= t:
                                angles_at_intervals.append(current_angle)
                                recorded_intervals.append(t)

                        # 記錄 0.00 到 0.50 秒每0.05秒的角度
                        for t in fine_interval_times:
                            if t not in fine_recorded_intervals and elapsed_time >= t:
                                fine_angles.append(current_angle)
                                fine_recorded_intervals.append(t)

                        # 檢查是否回到 pre_angle ± correction_threshold
                        if abs(current_angle - pre_angle) <= correction_threshold:
                            reaction_time = elapsed_time
                            final_angle = current_angle
                            print(f"已回到 pre_angle (±{correction_threshold} 度)，耗時 {reaction_time:.2f} 秒，最終角度: {final_angle:.2f} 度")

                        # 檢查是否超過3秒
                        if elapsed_time >= max_correction_time:
                            reaction_time = ">5s"
                            final_angle = current_angle
                            print("校正時間超過 3 秒，記錄反應時間為 >5s")
                            break

                        time.sleep(0.01)

                    # 確保記錄 0.5 秒內所有角度
                    while len(angles_at_intervals) < 5:
                        angles_at_intervals.append(current_angle)
                    while len(fine_angles) < 11:
                        fine_angles.append(current_angle)

                    # 計算 Delta
                    delta = final_angle - start_angle

                    # 計算 Reaction Time/Delta
                    reaction_time_per_delta = reaction_time / abs(delta) if isinstance(reaction_time, (int, float)) and delta != 0 else 0

                    # 儲存記錄
                    log_entry = {
                        "前角度 (度)": pre_angle,
                        "起始角度 (度)": start_angle,
                        "偏移角度 (度)": offset_angle,
                        "最終角度 (度)": final_angle,
                        "FAIL_OFFSET": fail_offset,
                        "反應時間 (秒)": reaction_time,
                        "Delta (度)": delta,
                        "反應時間/Delta (秒/度)": reaction_time_per_delta,
                        "0.1秒角度": angles_at_intervals[0],
                        "0.2秒角度": angles_at_intervals[1],
                        "0.3秒角度": angles_at_intervals[2],
                        "0.4秒角度": angles_at_intervals[3],
                        "0.5秒角度": angles_at_intervals[4],
                        "": "",  # 空白欄
                        "0.00秒角度": fine_angles[0],
                        "0.05秒角度": fine_angles[1],
                        "0.10秒角度": fine_angles[2],
                        "0.15秒角度": fine_angles[3],
                        "0.20秒角度": fine_angles[4],
                        "0.25秒角度": fine_angles[5],
                        "0.30秒角度": fine_angles[6],
                        "0.35秒角度": fine_angles[7],
                        "0.40秒角度": fine_angles[8],
                        "0.45秒角度": fine_angles[9],
                        "0.50秒角度": fine_angles[10],
                    }
                    log_data.append(log_entry)
                    print(f"記錄條目: {log_entry}")

                    # 將記錄寫入 Excel
                    df = pd.DataFrame(log_data)
                    df.to_excel(excel_filename, index=False)
                    print(f"數據已保存至 {excel_filename}")


                    
                    straight_start_time = None
                time.sleep(0.01)
                continue

            state = controller.get_state_engines(0).contents
            current_lx = state.lX
            current_angle = (current_lx / 32768) * (steering_range / 2)
            throttle_value = state.lY / 32768.0  # 獲取油門值
            is_straight = abs(current_angle) <= straight_threshold
            pre_angle = current_angle  # 記錄當前角度作為 pre_angle

            # 檢查油門狀態
            if throttle_value < throttle_threshold:
                if last_throttle_press_time == 0:
                    last_throttle_press_time = current_time
                    print(f"油門被踩下，值: {throttle_value:.2f}，開始計時")
            else:
                last_throttle_press_time = 0  # 重置油門計時
                straight_start_time = None  # 重置直行計時
                print(f"油門放開，值: {throttle_value:.2f}，重置計時")
                time.sleep(0.01)
                continue

            if is_straight and last_throttle_press_time != 0 and (current_time - last_throttle_press_time >= throttle_delay):
                if straight_start_time is None:
                    straight_start_time = current_time
                    print(f"檢測到直行，角度: {current_angle:.2f} 度，油門值: {throttle_value:.2f}")
                else:
                    straight_duration = current_time - straight_start_time
                    if straight_duration >= straight_duration_min and straight_duration <= straight_duration_max:
                        print(f"直行 {straight_duration:.2f} 秒，觸發轉向")
                        # 根據當前角度決定偏移方向
                        direction = 1 if current_angle >= 0 else -1
                        direction_str = "左" if direction == -1 else "右"
                        lx_change = (change_degree / (steering_range / 2)) * 32768
                        target_lx = current_lx + (lx_change * direction)
                        target_offset = (target_lx / 32768) * 100
                        target_offset = max(min(target_offset, 100), -100)
                        print(f"施加彈簧力向{direction_str}轉 {change_degree} 度 (偏移: {target_offset:.2f})")
                        controller.LogiPlaySpringForce(0, int(target_offset), force_magnitude, force_slope)
                        controller.LogiPlayDamperForce(0, 90)
                        controller.logi_update()
                        applying_force = True
                        force_start_time = current_time
                        straight_start_time = None
            else:
                if straight_start_time is not None:
                    print(f"非直行狀態，角度: {current_angle:.2f} 度，或油門未達延遲，重置計時")
                    straight_start_time = None

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n停止隨機轉向測試...")
        controller.LogiStopSpringForce(0)
        controller.LogiStopDamperForce(0)
        controller.logi_update()
        controller.steering_shutdown()
        if log_data:
            df = pd.DataFrame(log_data)
            df.to_excel(excel_filename, index=False)
            print(f"最終數據已保存至 {excel_filename}")
        print("測試已停止。")

def random_turn_test():
    controller = LogitechController()
    if not controller.steering_initialize():
        print("無法初始化控制器！")
        return
    random_turn_controller(controller)

if __name__ == "__main__":
    random_turn_test()