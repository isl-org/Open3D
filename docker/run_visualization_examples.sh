#!/usr/bin/bash
run_visualization_examples() {
    echo "Running Python visualization examples..."
    if [ "${DISPLAY:-unset}" = "unset" ]; then
        export DISPLAY=:71
        echo Starting Xvfb on display $DISPLAY
        Xvfb $DISPLAY &
    else
        echo Using DISPLAY=${DISPLAY}
    fi
    while IFS=, read -r example environment wait_time_s keyinput; do
        success=0
        if [[ "$example" = "#"* ]]; then
            continue
        fi
        printf "\nRunning example: $example\n"
        set -x
        env $environment DISPLAY=$DISPLAY timeout -v 12 open3d example $example &
        set +x
        sleep ${wait_time_s:-1}
        if [ "${keyinput:-}" != "" ]; then
            set -x
            xdotool search --limit 1 --name "Open3D" key --clearmodifiers --delay 1000 $keyinput
            set +x
        fi
        wait $! # Exit with exit code of last background process
        success=$?
        if [ $success != 0 ]; then
            printf >&2 "\n\n $example failed! \n\n"
        fi
    done <<CSVEOF
#example_name,environment,wait_time_s,keyinput
visualization/all_widgets,,3,Escape
visualization/add_geometry,,3,Escape
# crash in macOS
visualization/customized_visualization,,3,Escape
# crash in macOS
visualization/customized_visualization_key_action,,3,space Escape
visualization/demo_scene,,20,Escape
visualization/draw,,3,Escape Escape Escape Escape Escape
# visualization/draw_webrtc,,,Escape
# needs build options / deprecated
# visualization/headless_rendering,,,
# needs point picking
#visualization/interactive_visualization,,,y y k f Escape
visualization/line_width,,3,Escape
visualization/load_save_viewpoint,,3,Escape Escape
# needs data
# visualization/mitsuba_material_estimation,,,
visualization/mouse_and_point_coord,,3,Escape
# seg fault
visualization/multiple_windows,,3,Escape
visualization/non_blocking_visualization,,3,
visualization/non_english,,3,Escape
visualization/remove_geometry,,15,
visualization/render_to_image,EGL_PLATFORM=surfaceless,5,
visualization/text3d,,3,Escape Escape
# needs data
# visualization/textured_mesh,,,
# needs data
# visualization/textured_model,,,
# visualization/to_mitsuba,,,
visualization/video,,3,Escape
visualization/vis_gui,,3,Escape
CSVEOF
}
