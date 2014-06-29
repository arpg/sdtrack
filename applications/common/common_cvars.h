#pragma once

static bool& draw_landmarks =
    CVarUtils::CreateCVar<>("gui.DrawLandmarks", true, "");
static int& min_lm_measurements_for_drawing =
    CVarUtils::CreateCVar<>("gui.MinLmMeasurementsForDrawing", 2, "");
static int& selected_track_id =
    CVarUtils::CreateCVar<>("gui.SelectedTrackId", -1, "");
