def tune_focus() -> None:
    """Opens a window with a focus slider to find the correct focus value.

    Raises:
        RuntimeError: If the camera has not been initialized.
    """
    if cap is None:
        raise RuntimeError('Camera not initialized, call initialize() first.')

    def on_change(value: int) -> None:
        cap.set(cv2.CAP_PROP_FOCUS, value)

    cv2.namedWindow('Focus Tuning')
    cv2.createTrackbar(
        'Focus',
        'Focus Tuning',
        CALIBRATION_CONFIG['focus_value'],
        255,
        on_change,
    )
    while True:
        frame = get_frame()
        cv2.imshow('Focus Tuning', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()