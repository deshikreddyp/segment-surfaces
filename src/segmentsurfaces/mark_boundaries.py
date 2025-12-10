import numpy as np
import gmsh
import json
import argparse
import os
from collections import defaultdict, deque
from tqdm import tqdm


def angle_between(n1, n2, eps=1e-14):
    '''
    Returns angle between two vectors n1, n2 (need not be unit vectors).
    The angle units will be in radians.
    '''
    n1 = np.asarray(n1, dtype=float)
    n2 = np.asarray(n2, dtype=float)

    l1 = np.linalg.norm(n1)
    l2 = np.linalg.norm(n2)

    if l1 < eps or l2 < eps:
        return np.pi

    n1 = n1 / l1
    n2 = n2 / l2

    dotp = float(np.dot(n1,n2))
    if dotp > 1.0:
        dotp = 1.0
    if dotp < -1.0:
        dotp = -1.0

    return np.arccos(abs(dotp))


def compute_face_normals(faces):
    '''
    Returns list of facet normals corresponding to a list of faces.
    The way we input faces in the main script is that faces is a python list of tuples
    in the format [(dim_i, tag_i)] (although dim_i=2) for the type 'face'.
    '''
    face_normals = {}
    for dim, tag in faces:
        (u_min, v_min), (u_max, v_max) = gmsh.model.getParametrizationBounds(2, tag)
        u_mid = 0.5 * (u_min + u_max)
        v_mid = 0.5 * (v_min + v_max)

        nx, ny, nz = gmsh.model.getNormal(tag, [u_mid, v_mid])[0:3]
        n = np.array([nx, ny, nz], dtype=float)
        n_norm = np.linalg.norm(n)
        if n_norm > 0:
            n /= n_norm
        face_normals[tag] = n
    return face_normals


def build_face_neighbors(faces):
    '''
    face_tag -> list of neighboring face_tags
    Returns list of neighbouring face IDs to the given face tag.
    '''
    curve_to_faces = defaultdict(list) # key will be curve id. value will be a list of all face tags that touch this curve.

    for dim, tag in faces:
        bents = gmsh.model.getBoundary([(dim, tag)], oriented=False)
        for bdim, btag in bents:
            if bdim == 1:  # curve
                curve_to_faces[btag].append(tag) # add the face tag to the curve neighbour list in the dictionary

    face_neighbors = defaultdict(set) # because we only unique unordered pairs,
    # we go with set rather than lists
    for curve, flist in curve_to_faces.items():
        if len(flist) < 2:
            continue
        for i in range(len(flist)):
            for j in range(i + 1, len(flist)):
                fi, fj = flist[i], flist[j]
                face_neighbors[fi].add(fj)
                face_neighbors[fj].add(fi)

    return face_neighbors


def group_faces_by_normals_mask(faces, face_normals, face_neighbors, threshold_deg=30.0):
    """
    Same idea as group_faces_by_normals, but uses a 'mask queue' instead of deque.
    - faces: list of (2, tag)
    - face_normals: dict[tag] -> normal (np.array)
    - face_neighbors: dict[tag] -> set of neighbor tags
    """
    threshold = np.deg2rad(threshold_deg)
    face_groups = {}
    current_group = 0

    # Extract tags in a fixed order; this order determines the processing order
    face_tags = [tag for (_, tag) in faces]

    # Queue mask: tag -> bool (True means "in queue")
    queue_mask = {tag: False for tag in face_tags}

    for start_tag in face_tags:
        if start_tag in face_groups:
            continue

        current_group += 1

        # Seed the mask-queue with the starting face
        queue_mask[start_tag] = True

        # Process until no faces remain in the queue
        while True:
            # Find all faces currently in the queue
            active_queue = [t for t in face_tags if queue_mask[t]]
            if not active_queue:
                break

            # Take the first face in the queue according to face_tags order
            f = active_queue[0]
            queue_mask[f] = False  # remove from queue

            # If already assigned (might happen if multiple parents push same neighbor)
            if f in face_groups:
                continue

            face_groups[f] = current_group
            Nf = face_normals[f]

            # Check neighbors
            for nb in face_neighbors[f]:
                if nb in face_groups:
                    continue
                if angle_between(Nf, face_normals[nb]) < threshold:
                    # Mark neighbor into the queue (if not already there)
                    queue_mask[nb] = True

    print(f"Number of face groups found based on dihedral angle (mask version): {current_group}")
    return face_groups


def group_faces_by_normals(faces, face_normals, face_neighbors, threshold_deg=30.0):
    '''
    Sort of the main program. Loop over all the facets, and see till
    we dissatisfy threshold dihedral angle, and patch the group with
    a newer tag.
    '''
    threshold = np.deg2rad(threshold_deg)
    face_groups = {}
    current_group = 0

    face_tags = [tag for (_, tag) in faces]

    for tag in face_tags:
        if tag in face_groups:
            continue

        current_group += 1
        dq = deque([tag]) # this is the que where the remaining facets
        # that are about to tag are waiting for analysis
        face_groups[tag] = current_group

        while dq:  # loop till the que becomes empty
            f = dq.popleft()
            Nf = face_normals[f]
            for nb in face_neighbors[f]:
                if nb in face_groups:
                    continue
                if angle_between(Nf, face_normals[nb]) < threshold:
                    face_groups[nb] = current_group
                    dq.append(nb)

    print(f"Number of face groups found based on dihedral angle: {current_group}")
    return face_groups


def merge_face_groups(face_groups, face_neighbors, face_normals,
                      min_faces=None, max_facegroups=None):
    '''
    Post-processing: merge groups based on:
      1. min_faces: any group with < min_faces is merged into best neighbor.
      2. max_facegroups: keep merging smallest groups until we have <= max_facegroups.
    '''
    if min_faces is not None and max_facegroups is not None:
         print("Warning: Both min_faces and max_facegroups specified. This might conflict, but proceeding.")
         # Usually we might want to prioritize one, or just error out. 
         # mark_boundaries.py raises ValueError. Let's do that for safety if user wants strict parity.
         raise ValueError("Only one of min_faces or max_facegroups should be specified.")
         
    # ---------------------------------------------------------
    # Logic 1: min_faces
    # ---------------------------------------------------------
    if min_faces is not None:
        print(f">> Merging boundaries with less than {min_faces} faces")
        # We iterate until no group is smaller than min_faces or no changes possible
        while True:
            changed = False
            
            # Recompute group membership
            group_members = defaultdict(list)
            for f, g in face_groups.items():
                group_members[g].append(f)
            
            # Identify small groups
            small_groups = [g for g, flist in group_members.items() if len(flist) < min_faces]
            
            if not small_groups:
                break
                
            for g_small in small_groups:
                faces_small = group_members[g_small]
                # If it was already merged in this pass (rare but possible order dependence), check again
                if not faces_small: 
                    continue

                # Compute average normal of this small group
                ns = np.array([face_normals[f] for f in faces_small])
                avg_small = ns.mean(axis=0)
                if np.linalg.norm(avg_small) > 0:
                    avg_small /= np.linalg.norm(avg_small)

                # Find valid neighbor groups
                neighbor_groups = set()
                for f in faces_small:
                    for nb in face_neighbors[f]:
                        g_nb = face_groups[nb]
                        if g_nb != g_small:
                            neighbor_groups.add(g_nb)
                
                if not neighbor_groups:
                    continue # Isolated, can't merge

                # Find best neighbor (closest normal)
                best_g = None
                best_angle = None
                
                for g_nb in neighbor_groups:
                    # We need the avg normal of the neighbor group.
                    # Note: this is expensive to recompute inside the loop if groups are large.
                    # But consistent with mark_boundaries.py logic.
                    faces_nb = [Fx for Fx, Gx in face_groups.items() if Gx == g_nb] # slow lookup, optimized below
                    # Optimizing lookup:
                    faces_nb = group_members[g_nb]

                    nn = np.array([face_normals[f] for f in faces_nb])
                    avg_nb = nn.mean(axis=0)
                    if np.linalg.norm(avg_nb) > 0:
                        avg_nb /= np.linalg.norm(avg_nb)
                    
                    ang = angle_between(avg_small, avg_nb)
                    if best_angle is None or ang < best_angle:
                        best_angle = ang
                        best_g = g_nb
                
                if best_g is not None:
                    # Merge g_small into best_g
                    for f in faces_small:
                        face_groups[f] = best_g
                    # Update local view for next iteration
                    group_members[best_g].extend(faces_small)
                    del group_members[g_small]
                    changed = True
            
            if not changed:
                break

    # ---------------------------------------------------------
    # Logic 2: max_facegroups
    # ---------------------------------------------------------
    elif max_facegroups is not None:
        print(f">> Merging boundaries until total groups equal to {max_facegroups}")
        
        # Initial stats
        unique_groups = set(face_groups.values())
        initial_count = len(unique_groups)
        merges_needed = initial_count - max_facegroups
        
        if merges_needed > 0:
            pbar = tqdm(total=merges_needed, desc="Merging face groups")
            
            while True:
                # Re-evaluate current groups
                current_groups = set(face_groups.values())
                num_groups = len(current_groups)
                if num_groups <= max_facegroups:
                    break
                
                # Build group members dict
                group_members = defaultdict(list)
                for f, g in face_groups.items():
                    group_members[g].append(f)
                
                # Find smallest group
                # (group_id, size)
                sorted_groups = sorted([(g, len(flist)) for g, flist in group_members.items()], key=lambda x: x[1])
                g_small = sorted_groups[0][0]
                faces_small = group_members[g_small]
                
                # Average normal of small group
                ns = np.array([face_normals[f] for f in faces_small])
                avg_small = ns.mean(axis=0)
                if np.linalg.norm(avg_small) > 0:
                    avg_small /= np.linalg.norm(avg_small)
                
                # Neighbors
                neighbor_groups = set()
                for f in faces_small:
                    for nb in face_neighbors[f]:
                        g_nb = face_groups[nb]
                        if g_nb != g_small:
                            neighbor_groups.add(g_nb)
                            
                if not neighbor_groups:
                    # Can't merge isolated group. MUST break to avoid infinite loop
                    # strictly speaking we might have other small groups we COULD merge,
                    # but the logic says "merge the smallest". If the smallest is stuck, we're stuck.
                    print("Warning: Smallest group is isolated, cannot merge further.")
                    break

                best_g = None
                best_angle = None
                
                for g_nb in neighbor_groups:
                    faces_nb = group_members[g_nb]
                    nn = np.array([face_normals[f] for f in faces_nb])
                    avg_nb = nn.mean(axis=0)
                    if np.linalg.norm(avg_nb) > 0:
                        avg_nb /= np.linalg.norm(avg_nb)
                    
                    ang = angle_between(avg_small, avg_nb)
                    if best_angle is None or ang < best_angle:
                        best_angle = ang
                        best_g = g_nb
                
                if best_g is not None:
                    # Merge
                    for f in faces_small:
                        face_groups[f] = best_g
                    pbar.update(1)
                else:
                    break
            
            pbar.close()

    # Refactor / Renumber
    old_to_new = {}
    new_id = 0
    for g_old in sorted(set(face_groups.values())):
        new_id += 1
        old_to_new[g_old] = new_id

    for f, g in list(face_groups.items()):
        face_groups[f] = old_to_new[g]

    print(f"Final number of face groups after merging: {len(set(face_groups.values()))}")
    return face_groups


def create_physical_groups_from_face_groups(face_groups, name_prefix="Bnd"):
    '''
    All of this is a gmsh thingy. This is required to store the physical tags.
    GMSH is a opensource platform, whihc most researchers in computational modeling
    community use.
    '''
    groups_to_faces = defaultdict(list)
    for f, g in face_groups.items():
        groups_to_faces[g].append(f)

    for g, flist in groups_to_faces.items():
        if not flist:
            continue
        pg_tag = gmsh.model.addPhysicalGroup(2, flist, tag=g)
        gmsh.model.setPhysicalName(2, pg_tag, f"{name_prefix}_{g}")
    print(f"Created {len(groups_to_faces)} Physical Surface groups (labels).")
    print(f"Total number of labels created: {len(groups_to_faces)}")


def export_brep_and_mapping(brep_out_path, mapping_path, face_groups):
    gmsh.write(brep_out_path)
    print(f"Exported geometry to: {brep_out_path}")

    with open(mapping_path, "w") as f:
        json.dump(face_groups, f, indent=2)
    print(f"Exported face_groups mapping to: {mapping_path}")


def main():
    parser = argparse.ArgumentParser(description="Segment a BREP file into face groups based on dihedral angles.")
    parser.add_argument("input_brep", help="Path to the input .brep file")
    parser.add_argument("--output", help="Path to the output file (e.g. output.brep, output.msh, output.geo_unrolled)", default=None)
    parser.add_argument("--output_json", help="Path to the output face groups JSON map (only used if output is .brep)", default=None)
    parser.add_argument("--threshold", type=float, default=30.0, help="Dihedral angle threshold in degrees (default: 30.0)")
    parser.add_argument("--min_faces", type=int, default=None, help="Merge boundaries with less than MIN_FACES faces")
    parser.add_argument("--max_facegroups", type=int, default=None, help="Merge boundaries until MAX_FACEGROUPS groups remain")
    
    args = parser.parse_args()
    
    # Validation
    if args.min_faces is not None and args.max_facegroups is not None:
        parser.error("Cannot specify both --min_faces and --max_facegroups")

    in_brep = args.input_brep
    
    # Generate default output paths if not provided
    base_name, _ = os.path.splitext(in_brep)
    if args.output:
        out_path = args.output
    else:
        out_path = f"{base_name}_tagged.brep"
        
    # Logic for JSON path defaults
    if args.output_json:
        map_json = args.output_json
    else:
        map_json = f"{base_name}_face_groups.json"

    # Load .brep file
    gmsh.initialize()
    gmsh.model.add("brep_model")
    try:
        gmsh.model.occ.importShapes(in_brep)
    except Exception as e:
        print(f"Error loading {in_brep}: {e}")
        gmsh.finalize()
        return

    gmsh.model.occ.synchronize()
    print(f"Loaded BREP: {in_brep}")

    # Pull the faces (entity ID: 2) from the .brep file
    faces = gmsh.model.occ.getEntities(dim=2)  # [(2, tag), ...]
    print(f"Number of OCC faces: {len(faces)}")

    face_normals  = compute_face_normals(faces)
    face_neighbors = build_face_neighbors(faces)

    # Our first round of segmentation based on dihedral angles
    face_groups = group_faces_by_normals(
        faces, face_normals, face_neighbors,
        threshold_deg=args.threshold
    )
    # face_groups = group_faces_by_normals_mask(
    #     faces, face_normals, face_neighbors,
    #     threshold_deg=args.threshold
    # )

    # Optional merging
    face_groups = merge_face_groups(
        face_groups,
        face_neighbors,
        face_normals,
        min_faces=args.min_faces,
        max_facegroups=args.max_facegroups
    )

    create_physical_groups_from_face_groups(face_groups, name_prefix="Bnd")
    
    if args.dry_run:
        print("Dry run enabled: Skipping file export.")
    else:
        # Export based on extension
        _, ext = os.path.splitext(out_path)
        ext = ext.lower()
        
        if ext == ".brep":
            export_brep_and_mapping(out_path, map_json, face_groups)
        elif ext == ".vtk":
            print(f"Generating 2D mesh for {ext} export...")
            gmsh.option.setNumber("Mesh.SaveAll", 1)  # Ensure everything is saved
            gmsh.model.mesh.generate(2)
            gmsh.write(out_path)
            print(f"Exported mesh with tags to: {out_path}")
        else:
            print(f"Error: Unsupported export format '{ext}'. Only .brep and .vtk are supported.")

    gmsh.finalize()

if __name__ == "__main__":
    main()
