from omni.isaac.kit import SimulationApp
import os
import argparse
import glob

parser = argparse.ArgumentParser("Dataset generator")
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--height", type=int, default=544)
parser.add_argument("--width",  type=int, default=960)
parser.add_argument("--num_frames", type=int, default=7000)
parser.add_argument("--distractors", type=str, default="warehouse")
parser.add_argument("--data_dir", type=str, default=os.getcwd() + "/_all_colors_data")

args, unknown_args = parser.parse_known_args()

CONFIG = {
    "renderer": "PathTracing",
    "headless": args.headless,
    "width": args.width,
    "height": args.height,
    "num_frames": args.num_frames,
}

simulation_app = SimulationApp(launch_config=CONFIG)

import carb
import omni
import omni.usd
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import get_current_stage
from pxr import Usd, Sdf, UsdLux, Gf, Semantics   # un solo import
import omni.replicator.core as rep

rep.settings.carb_settings("/omni/replicator/RTSubframes", 4)

YELLOW_CUBE = ['file:///home/ferestrada/synth_yolo/usd/usd2/pallet_flat.usd']

ALL_CUBES = {
    "yellow": YELLOW_CUBE,
}

# Path del mesh hijo que quieres anotar DENTRO del USD
# (relativo a como Replicator lo instancia en /Replicator/Ref_Xform_XX/Ref/...)
# Ajusta "pallet_side1" al nombre real de tu prim hijo
CHILD_PRIM_SUFFIX = "pallet_side1"   # ← cambia esto al nombre correcto

CUBE_POSE_RANGES = {
    "yellow": ((-0.15, 1.10, 0.75), (0.15, 1.80, 1.25)),
}

CAM_POS_MIN = (-1.00, -1.50, 0.10)
CAM_POS_MAX = ( 1.00,  1.00, 1.20)

DISTRACTORS_WAREHOUSE = 10 * [
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
            for prop in prim.GetProperties():
                if not Semantics.SemanticsAPI.IsSemanticsAPIPath(prop.GetPath()):
                    continue
                instance_name = prop.SplitName()[1]
                if instance_name in processed_instances:
                    continue
                processed_instances.add(instance_name)
                sem = Semantics.SemanticsAPI.Get(prim, instance_name)
                if sem.GetSemanticDataAttr().Get() not in keep_semantics:
                    prim.RemoveProperty(sem.GetSemanticTypeAttr().GetName())
                    prim.RemoveProperty(sem.GetSemanticDataAttr().GetName())
                    prim.RemoveAPI(Semantics.SemanticsAPI, instance_name)


def fix_semantics_on_child(stage, label, child_suffix):
    """
    Replicator coloca el USD bajo /Replicator/Ref_Xform_XX/Ref/...
    Esta función busca todos los prims cuyo nombre coincida con child_suffix,
    quita la semántica del padre raíz y la pone en el hijo correcto.
    """
    for prim in stage.Traverse():
        if prim.GetName() == child_suffix:
            # Aplicar semántica en el hijo
            sem = Semantics.SemanticsAPI.Apply(prim, "class")
            sem.GetSemanticTypeAttr().Set("class")
            sem.GetSemanticDataAttr().Set(label)
            print(f"[SDG] Semántica '{label}' aplicada en: {prim.GetPath()}")


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


def add_all_cubes():
    groups = {}
    for label, usd_list in ALL_CUBES.items():
        rep_objs = [rep.create.from_usd(path, count=1) for path in usd_list]
        groups[label] = rep.create.group(rep_objs)
    return groups


def add_distractors(distractor_type="warehouse"):
    full_distractors = full_distractors_list(distractor_type)
    distractors = [rep.create.from_usd(p, count=1) for p in full_distractors]
    return rep.create.group(distractors)


def list_hdri_paths(folder):
    return sorted(glob.glob(os.path.join(folder, "*.exr")))


def random_hdri(hdri_paths):
    dome = rep.get.prims("/World/DomeLight")
    if dome != 0:
        dome = rep.create.light(
            light_type="Dome",
            name="DomeLight",
            rotation=(0, 0, 0),
            intensity=5000.0,
        )
    with dome:
        rep.modify.attribute("inputs:texture:file", rep.distribution.choice(hdri_paths))
        rep.modify.pose(rotation=rep.distribution.uniform((0, 0, 0), (0, 0, 360)))
        rep.modify.attribute("exposure", rep.distribution.uniform(-7.0, 0.0))


def main():
    hdri_path = '/home/ferestrada/synth_yolo/hdri/'
    hdris = list_hdri_paths(hdri_path)

    omni.usd.get_context().new_stage()
    stage = get_current_stage()

    for i in range(100):
        if i % 10 == 0:
            print(f"App update {i}..")
        simulation_app.update()

    cube_groups = add_all_cubes()
    rep_distractor_group = add_distractors(distractor_type=args.distractors)

    for _ in range(10):
        simulation_app.update()


    update_semantics(stage=stage, keep_semantics=[])          # limpia todo
    for label in ALL_CUBES.keys():
        fix_semantics_on_child(stage, label, CHILD_PRIM_SUFFIX)

    cam = rep.create.camera(clipping_range=(0.1, 1000000))
    RESOLUTION = (CONFIG["width"], CONFIG["height"])
    render_product = rep.create.render_product(cam, RESOLUTION)

    all_cube_prims = rep.create.group(list(cube_groups.values()))

    with rep.trigger.on_frame(num_frames=CONFIG["num_frames"]):

        random_hdri(hdris)

        for label, group in cube_groups.items():
            pos_min, pos_max = CUBE_POSE_RANGES[label]
            with group:
                rep.modify.pose(
                    position=rep.distribution.uniform(pos_min, pos_max),
                    rotation=rep.distribution.uniform((0, 0, 0), (10, 10, 360)),
                    scale=(2.4, 2.4, 2.4),
                )


                rep.randomizer.materials( #using only blue 
                    rep.create.material_omnipbr(
                        diffuse=rep.distribution.uniform(
                            (0.0, 0.0, 0.4),   
                            (0.0, 0.1, 1.0),  
                        ),
                        roughness=rep.distribution.uniform(0.2, 2.0),
                        metallic=rep.distribution.uniform(0.0, 0.3),
                        count=1,
                    )
                )

        with cam:
            rep.modify.pose(
                position=rep.distribution.uniform(CAM_POS_MIN, CAM_POS_MAX),
                look_at=all_cube_prims,
            )
            rep.modify.attribute("focalLength", rep.distribution.uniform(18.0, 50.0))
            rep.modify.attribute("fStop", rep.distribution.uniform(600.0, 620.0))

        if args.distractors != "None":
            with rep_distractor_group:
                rep.modify.pose(
                    position=rep.distribution.uniform((-10, 0.5, 0), (10, 6, 0)),
                    rotation=rep.distribution.uniform((0, 0, 0), (0, 0, 360)),
                    scale=rep.distribution.uniform(1, 1.5),
                )

    output_directory = args.data_dir
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir=output_directory,
        rgb=True,
        instance_segmentation=True,
        semantic_segmentation=True,
        bounding_box_2d_tight=False,
        bounding_box_2d_loose=False,
    )
    writer.attach([render_product])

    rep.orchestrator.run()
    while not rep.orchestrator.get_is_started():
        simulation_app.update()
    while rep.orchestrator.get_is_started():
        simulation_app.update()
        rep.BackendDispatch.wait_until_done()
    rep.orchestrator.stop()
    simulation_app.close()
'''
    while True:
        simulation_app.update()
'''

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        carb.log_error(f"Exception: {e}")
        import traceback
        traceback.print_exc()
        print('ERROR')