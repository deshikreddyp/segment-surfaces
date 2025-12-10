import gmsh
import json
import argparse
import sys
import os
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Load a BREP file and apply Physical Groups from a JSON mapping.")
    parser.add_argument("input_brep", help="Path to the input .brep file")
    parser.add_argument("input_json", help="Path to the input .json mapping file")
    parser.add_argument("--nopopup", action="store_true", help="Do not open the Gmsh GUI")

    args = parser.parse_args()

    if not os.path.exists(args.input_brep):
        print(f"Error: BREP file not found: {args.input_brep}")
        sys.exit(1)
    if not os.path.exists(args.input_json):
        print(f"Error: JSON file not found: {args.input_json}")
        sys.exit(1)

    # Initialize Gmsh
    gmsh.initialize()
    gmsh.model.add("loaded_model")

    # Load Geometry
    try:
        gmsh.model.occ.importShapes(args.input_brep)
        gmsh.model.occ.synchronize()
    except Exception as e:
        print(f"Error importing BREP: {e}")
        gmsh.finalize()
        sys.exit(1)

    print(f"Loaded geometry from: {args.input_brep}")

    # Load JSON Mapping
    try:
        with open(args.input_json, 'r') as f:
            face_groups = json.load(f)
    except Exception as e:
        print(f"Error reading JSON: {e}")
        gmsh.finalize()
        sys.exit(1)

    print(f"Loaded mapping from: {args.input_json}")

    # Group faces by their group ID
    # JSON keys are strings, values are integers (Group IDs)
    groups_to_faces = defaultdict(list)
    for face_tag_str, group_id in face_groups.items():
        face_tag = int(face_tag_str)
        groups_to_faces[group_id].append(face_tag)

    # Create Physical Groups
    count = 0
    for group_id, faces in groups_to_faces.items():
        if not faces:
            continue
        # Check if faces exist in model? 
        # Gmsh might error if we try to group a tag that doesn't exist, 
        # but if the BREP matches the JSON, it should be fine.
        try:
            pg_tag = gmsh.model.addPhysicalGroup(2, faces, tag=group_id)
            gmsh.model.setPhysicalName(2, pg_tag, f"Bnd_{group_id}")
            count += 1
        except Exception as e:
            print(f"Warning: Could not create group {group_id} for faces {faces}: {e}")

    print(f"Successfully created {count} Physical Surface groups.")

    # Launch GUI
    if not args.nopopup:
        print("Launching Gmsh GUI...")
        gmsh.fltk.run()

    gmsh.finalize()

if __name__ == "__main__":
    main()
