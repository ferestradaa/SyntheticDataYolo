from omni.isaac.kit import SimulationApp

import json
import os
import sys
import argparse
import glob
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_config, resolve, DEFAULT_CONFIG

parser = argparse.ArgumentParser("Dataset generator")
parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
parser.add_argument("--headless", action="store_true", default=True, help="No window")
parser.add_argument("--height", type=int)   # height, width, num_frames, distractors and data_dir override the config
parser.add_argument("--width",  type=int)
parser.add_argument("--num_frames", type=int)
parser.add_argument("--distractors", type=str)
parser.add_argument("--data_dir", type=str)

args, unknown_args = parser.parse_known_args()

cfg = load_config(args.config)
for key in ("height", "width", "num_frames", "distractors", "data_dir"):
    if getattr(args, key) is None:
        setattr(args, key, cfg[key])

CONFIG = {"renderer": "PathTracing", "headless": args.headless,
          "width": args.width, "height": args.height, "num_frames": args.num_frames}

simulation_app = SimulationApp(launch_config=CONFIG)

import carb
import omni
import omni.usd
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import get_current_stage
from pxr import Semantics
import omni.replicator.core as rep
import numpy as np
from pxr import Usd, UsdGeom, Sdf, UsdLux, Gf


import domain_randomization as dr

rep.settings.carb_settings("/omni/replicator/RTSubframes", 4)
if cfg.get("seed") is not None:
    rep.set_global_seed(cfg["seed"])

RT_SUBFRAMES = 4

ASSET_PATHS = cfg["assets"]
INSTANCES_PER_ASSET = cfg["instances_per_asset"]


DISTRACTORS_WAREHOUSE = 2 * [
    "/Isaac/Environments/Simple_Warehouse/Props/S_TrafficCone.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/S_WetFloorSign.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_A_01.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_A_02.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_A_03.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_B_01.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_B_03.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_C_02.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BottlePlasticA_02.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BottlePlasticB_01.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BottlePlasticD_01.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BottlePlasticE_01.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BucketPlastic_B.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxB_01_1262.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxB_01_1268.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxB_01_1482.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxB_01_1683.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxB_01_291.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_01_1454.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_01_1513.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CratePlastic_A_04.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CratePlastic_B_03.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CratePlastic_B_05.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CratePlastic_C_02.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CratePlastic_E_02.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_PushcartA_02.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_RackPile_04.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_RackPile_03.usd",
]

DISTRACTORS_ADDITIONAL = [
    "/Isaac/Environments/Hospital/Props/Pharmacy_Low.usd",
    "/Isaac/Environments/Hospital/Props/SM_BedSideTable_01b.usd",
    "/Isaac/Environments/Hospital/Props/SM_BooksSet_26.usd",
    "/Isaac/Environments/Hospital/Props/SM_BottleB.usd",
    "/Isaac/Environments/Hospital/Props/SM_BottleA.usd",
    "/Isaac/Environments/Hospital/Props/SM_BottleC.usd",
    "/Isaac/Environments/Hospital/Props/SM_Cart_01a.usd",
    "/Isaac/Environments/Hospital/Props/SM_Chair_02a.usd",
    "/Isaac/Environments/Hospital/Props/SM_Chair_01a.usd",
    "/Isaac/Environments/Hospital/Props/SM_Computer_02b.usd",
    "/Isaac/Environments/Hospital/Props/SM_Desk_04a.usd",
    "/Isaac/Environments/Hospital/Props/SM_DisposalStand_02.usd",
    "/Isaac/Environments/Hospital/Props/SM_FirstAidKit_01a.usd",
    "/Isaac/Environments/Hospital/Props/SM_GasCart_01c.usd",
    "/Isaac/Environments/Hospital/Props/SM_Gurney_01b.usd",
    "/Isaac/Environments/Hospital/Props/SM_HospitalBed_01b.usd",
    "/Isaac/Environments/Hospital/Props/SM_MedicalBag_01a.usd",
    "/Isaac/Environments/Hospital/Props/SM_Mirror.usd",
    "/Isaac/Environments/Hospital/Props/SM_MopSet_01b.usd",
    "/Isaac/Environments/Hospital/Props/SM_SideTable_02a.usd",
    "/Isaac/Environments/Hospital/Props/SM_SupplyCabinet_01c.usd",
    "/Isaac/Environments/Hospital/Props/SM_SupplyCart_01e.usd",
    "/Isaac/Environments/Hospital/Props/SM_TrashCan.usd",
    "/Isaac/Environments/Hospital/Props/SM_Washbasin.usd",
    "/Isaac/Environments/Hospital/Props/SM_WheelChair_01a.usd",
    "/Isaac/Environments/Office/Props/SM_WaterCooler.usd",
    "/Isaac/Environments/Office/Props/SM_TV.usd",
    "/Isaac/Environments/Office/Props/SM_TableC.usd",
    "/Isaac/Environments/Office/Props/SM_Recliner.usd",
    "/Isaac/Environments/Office/Props/SM_Personenleitsystem_Red1m.usd",
    "/Isaac/Environments/Office/Props/SM_Lamp02_162.usd",
    "/Isaac/Environments/Office/Props/SM_Lamp02.usd",
    "/Isaac/Environments/Office/Props/SM_HandDryer.usd",
    "/Isaac/Environments/Office/Props/SM_Extinguisher.usd",
]

TEXTURES = [
    "/Isaac/Materials/Textures/Patterns/nv_asphalt_yellow_weathered.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_tile_hexagonal_green_white.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_rubber_woven_charcoal.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_granite_tile.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_tile_square_green.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_marble.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_brick_reclaimed.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_concrete_aged_with_lines.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_wooden_wall.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_stone_painted_grey.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_wood_shingles_brown.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_tile_hexagonal_various.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_carpet_abstract_pattern.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_wood_siding_weathered_green.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_animalfur_pattern_greys.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_artificialgrass_green.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_bamboo_desktop.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_brick_red_stacked.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_fireplace_wall.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_fabric_square_grid.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_gravel_grey_leaves.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_plastic_blue.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_stone_red_hatch.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_stucco_red_painted.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_stucco_smooth_blue.jpg",
]

def update_semantics(stage, keep_semantics=[]):
    for prim in stage.Traverse():
        if prim.HasAPI(Semantics.SemanticsAPI):
            processed_instances = set()
            for property in prim.GetProperties():
                is_semantic = Semantics.SemanticsAPI.IsSemanticsAPIPath(property.GetPath())
                if is_semantic:
                    instance_name = property.SplitName()[1]
                    if instance_name in processed_instances:
                        continue
                    processed_instances.add(instance_name)
                    sem = Semantics.SemanticsAPI.Get(prim, instance_name)
                    type_attr = sem.GetSemanticTypeAttr()
                    data_attr = sem.GetSemanticDataAttr()
                    if data_attr.Get() not in keep_semantics:
                        prim.RemoveProperty(type_attr.GetName())
                        prim.RemoveProperty(data_attr.GetName())
                        prim.RemoveAPI(Semantics.SemanticsAPI, instance_name)


def prefix_with_isaac_asset_server(relative_path):
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        raise Exception("Nucleus server not found")
    return assets_root_path + relative_path


def full_distractors_list(distractor_type="warehouse"):
    full_dist_list = []
    if distractor_type == "warehouse":
        for d in DISTRACTORS_WAREHOUSE:
            full_dist_list.append(prefix_with_isaac_asset_server(d))
    elif distractor_type == "additional":
        for d in DISTRACTORS_ADDITIONAL:
            full_dist_list.append(prefix_with_isaac_asset_server(d))
    else:
        print("No Distractors being added")
    return full_dist_list


def full_textures_list():
    return [prefix_with_isaac_asset_server(t) for t in TEXTURES]


def load_assets():
    assets = {}
    for entry in ASSET_PATHS:
        rel, lying = (entry, 0.0) if isinstance(entry, str) else (entry["path"], entry.get("lying", 0.0))
        path = resolve(rel).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Asset not found: {path}")
        stage = Usd.Stage.Open(str(path))
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        box = cache.ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange()
        assets[path.stem] = {"url": f"file://{path}", "bbox_min": np.array(box.GetMin()), "bbox_max": np.array(box.GetMax()), "lying": lying}
        print(f"[asset] {path.stem}: size={tuple(round(x, 3) for x in box.GetSize())} m, lying {lying}")
    return assets


def add_all_assets(assets):
    for label, asset in assets.items():
        rep.create.from_usd(asset["url"], semantics=[("class", label)], count=INSTANCES_PER_ASSET)


def add_distractors(distractor_type="warehouse"):
    full_distractors = full_distractors_list(distractor_type)
    distractors = [rep.create.from_usd(p, count=1) for p in full_distractors]
    return rep.create.group(distractors)


def list_hdri_paths(folder):
    return sorted(glob.glob(os.path.join(folder, "*.exr")))


def main():
    hdri_path = str(resolve('hdri'))
    hdris = list_hdri_paths(hdri_path)
    

    omni.usd.get_context().new_stage()
    stage = get_current_stage()

    for i in range(100):
        if i % 10 == 0:
            print(f"App update {i}..")
        simulation_app.update()

    if not rep.get.prims("/World/DomeLight").get_output_prims().get("prims", []):
        rep.create.light(light_type="Dome", name="DomeLight", rotation=(0, 0, 0), intensity=5000.0)   # the randomizer drives it per frame

    assets = load_assets()
    add_all_assets(assets)
    add_distractors(distractor_type=args.distractors)

    update_semantics(stage=stage, keep_semantics=list(assets.keys()))

    cam = rep.create.camera(clipping_range=(0.1, 1000000))
    RESOLUTION = (CONFIG["width"], CONFIG["height"])
    render_product = rep.create.render_product(cam, RESOLUTION)

    surfaces = dr.add_surfaces()
    textures = full_textures_list()
    for _ in range(5):
        simulation_app.update()   # let the referenced prims load so their bounds can be measured
    randomizer = dr.SceneRandomizer(stage, assets, RESOLUTION, hdris=hdris, seed=cfg.get("seed"))

    with rep.trigger.on_frame():
        dr.randomize_surfaces(surfaces, textures)   # material swapping stays in the graph, the dome is driven per frame from python

    output_directory = str(resolve(args.data_dir))
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir=output_directory,
        rgb=True,
        instance_segmentation=True,
        semantic_segmentation=False,
        bounding_box_2d_tight=False,
        bounding_box_2d_loose=False,
    )
    writer.attach([render_product])

    frames_meta = []
    for frame in range(args.num_frames):
        frames_meta.append(randomizer.randomize_frame())   # assets, camera, distractors and lights are placed here, the graph only does hdri and surface textures
        rep.orchestrator.step(rt_subframes=RT_SUBFRAMES)
    rep.orchestrator.wait_until_complete()
    rep.BackendDispatch.wait_until_done()
    rep.orchestrator.stop()

    with open(os.path.join(output_directory, "frames_meta.json"), "w") as f:   # what each frame was supposed to show, used by validate_dataset.py
        json.dump(frames_meta, f)
    dr.apply_camera_effects(output_directory, cfg.get("seed"))
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        carb.log_error(f"Exception: {e}")
        import traceback
        traceback.print_exc()
        print('ERROR')