from agile.rl_env.tasks.locomotion.t1.velocity_env_cfg import T1LowerVelocityEnvCfg
from agile.rl_env.tasks.pick_place.pick_place_tracking_env_cfg import PickPlaceTrackingEnvCfg


def test_t1_velocity_eval_uses_the_standard_clean_evaluation_configuration():
    cfg = T1LowerVelocityEnvCfg()

    cfg.eval()

    assert cfg.scene.terrain.terrain_type == "plane"
    assert cfg.scene.terrain.terrain_generator is None
    assert cfg.viewer.eye == (-2.5, -5.0, 2.0)
    assert cfg.viewer.lookat == (0.0, 0.0, 0.75)
    assert cfg.viewer.origin_type == "world"
    assert cfg.rewards is None
    assert cfg.curriculum is None
    assert cfg.events is not None
    assert cfg.scene.terrain.physics_material is None
    assert cfg.scene.terrain.visual_material is None
    assert cfg.sim.physics_material is not None


def test_pick_place_uses_beta2_compatible_plain_ground_plane():
    cfg = PickPlaceTrackingEnvCfg()

    assert cfg.scene.terrain.terrain_type == "plane"
    assert cfg.scene.terrain.physics_material is None
    assert cfg.scene.terrain.visual_material is None
    assert cfg.sim.physics_material is not None
