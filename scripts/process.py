import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from math import acos, degrees
import argparse

# Check for dependencies
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    sys.stderr.write("RDKit is required for this script.\n")
    sys.exit(1)

try:
    from pyscf import gto, scf
    from pyscf.geomopt.berny_solver import optimize
    from pyscf.hessian import thermo
except ImportError:
    sys.stderr.write("PySCF is required for this script.\n")
    sys.exit(1)

# Unit conversion constants
BOHR_TO_ANG = 0.529177  # 1 Bohr in Angstrom
HARTREE_TO_J = 4.359744e-18  # 1 Hartree in Joules
BOHR_TO_M = 5.29177e-11     # 1 Bohr in meters
# Conversion factor for force constants: Eh/Bohr^2 to mdyn/Å
factor_N_per_m = HARTREE_TO_J / (BOHR_TO_M**2)       # Hartree/Bohr^2 to N/m
factor_mdyn_per_A = factor_N_per_m * 0.01            # N/m to mdyn/Å (1 N/m = 0.01 mdyn/Å)

def const(smiles):
    # 1. SMILES to 3D geometry using RDKit
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        sys.stderr.write(f"Error: Failed to parse SMILES '{smiles}'.\n")
        sys.exit(1)
    # Add hydrogens and embed 3D coordinates
    mol = Chem.AddHs(mol)
    # Use ETKDG for better initial conformer generation
    params = AllChem.ETKDG()
    params.randomSeed = 1  # deterministic embedding for reproducibility
    embed_status = AllChem.EmbedMolecule(mol, params)
    if embed_status != 0:
        # Retry embedding if initial attempt fails
        embed_status = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if embed_status != 0:
        sys.stderr.write("Error: RDKit embedding failed.\n")
        sys.exit(1)
    # Optimize geometry with MMFF to relieve bad contacts
    AllChem.MMFFOptimizeMolecule(mol)

    # Extract atomic coordinates and symbols from RDKit molecule
    conf = mol.GetConformer()
    rd_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    rd_coords_ang = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=float)

    # 2. Setup PySCF molecule and run initial HF calculation
    # Use the coordinates from RDKit (in Angstrom)
    mol_pyscf = gto.Mole(atom=[(sym, tuple(coord)) for sym, coord in zip(rd_symbols, rd_coords_ang)],
                         unit="Angstrom", basis="6-311++G(d,p)")
    mol_pyscf.build()
    mf = scf.RHF(mol_pyscf)
    mf.conv_tol = 1e-9  # tighten convergence for accuracy
    mf.verbose = 0      # suppress output
    mf.kernel()         # run HF SCF

    # 3. Geometry optimization
    try:
        mol_eq = optimize(mf, maxsteps=100)  # returns optimized Mole object
    except Exception as e:
        sys.stderr.write(f"Optimization failed: {e}\n")
        sys.exit(1)
    # Final HF on optimized geometry
    mf_eq = scf.RHF(mol_eq).run(conv_tol=1e-9, verbose=0)

    # 4. Compute Hessian (analytic second derivatives)
    hessian = mf_eq.Hessian().kernel()

    # 5. Vibrational frequency analysis
    freq_info = thermo.harmonic_analysis(mol_eq, hessian, exclude_trans=True, exclude_rot=True, imaginary_freq=False)
    freqs_cm = freq_info["freq_wavenumber"]
    # Only take non-negative frequencies
    freqs_cm = np.real(freqs_cm)
    print("\nVibrational frequencies (cm^-1):")
    print(", ".join(f"{f:.1f}" for f in freqs_cm if f > 1e-2) or "None")

    # Convert optimized coordinates to convenient form
    coords_bohr = mol_eq.atom_coords(unit="Bohr")  # atomic coordinates in Bohr
    natm = mol_eq.natm

    # Helper function to compute internal coordinate force constant from Hessian
    def internal_force_constant(v_vec):
        """Compute force constant k = v^T H v for a given internal coordinate displacement vector v (in Bohr)."""
        k = 0.0
        for i in range(natm):
            for j in range(natm):
                k += np.dot(v_vec[i], hessian[i, j].dot(v_vec[j]))
        return k

    # 6. Extract and print optimized bond lengths and bond angles
    print("\nOptimized bond lengths:")
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        dist = np.linalg.norm(coords_bohr[j] - coords_bohr[i]) * BOHR_TO_ANG
        atom_i = mol.GetAtomWithIdx(i)
        atom_j = mol.GetAtomWithIdx(j)
        print(f"  Bond {atom_i.GetSymbol()}{i}-{atom_j.GetSymbol()}{j}: {dist:.3f} Å")

    print("\nOptimized bond angles:")
    angles_list = []
    for atom in mol.GetAtoms():
        idx_center = atom.GetIdx()
        neighbors = [nbr.GetIdx() for nbr in atom.GetNeighbors()]
        if len(neighbors) < 2:
            continue
        # Consider unique pairs of neighbors to form angles
        for a in range(len(neighbors)):
            for b in range(a+1, len(neighbors)):
                i = neighbors[a]
                k = neighbors[b]
                # Angle i-center-k
                vec1 = coords_bohr[i] - coords_bohr[idx_center]
                vec2 = coords_bohr[k] - coords_bohr[idx_center]
                # Compute angle in degrees
                cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                # Numerical safety: clamp within [-1,1]
                cos_angle = max(-1.0, min(1.0, cos_angle))
                angle_deg = degrees(acos(cos_angle))
                angles_list.append((i, idx_center, k, angle_deg))
                atom_i = mol.GetAtomWithIdx(i)
                atom_center = mol.GetAtomWithIdx(idx_center)
                atom_k = mol.GetAtomWithIdx(k)
                print(f"  Angle {atom_i.GetSymbol()}{i}-{atom_center.GetSymbol()}{idx_center}-{atom_k.GetSymbol()}{k}: {angle_deg:.1f}°")

    # 7. Mulliken atomic charges
    print("\nMulliken atomic charges:")
    pop, charges = mf_eq.mulliken_pop()  # Mulliken population analysis
    for idx, atom in enumerate(mol.GetAtoms()):
        print(f"  {atom.GetSymbol()}{idx}: {charges[idx]: .3f}")

    # 8. Force constants for bond stretching and angle bending
    print("\nForce constants:")
    # Bond stretching force constants
    bond_force = 0
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # Unit vector along bond (in Bohr)
        rij = coords_bohr[j] - coords_bohr[i]
        r = np.linalg.norm(rij)
        if r < 1e-8:
            continue
        u = rij / r
        # Internal coordinate displacement vector for bond length change
        v = np.zeros((natm, 3))
        v[i] -= 0.5 * u
        v[j] += 0.5 * u
        k_au = internal_force_constant(v)      # in Hartree/Bohr^2
        bond_force = k_au   # convert to mdyn/Å
        atom_i = mol.GetAtomWithIdx(i)
        atom_j = mol.GetAtomWithIdx(j)
        print(f"  Bond {atom_i.GetSymbol()}{i}-{atom_j.GetSymbol()}{j} force constant: {bond_force *1556.90013 } kj/A^2")

    # Angle bending force constants
    angle_k = 0
    for (i, j, k, angle_deg) in angles_list:
        # Vectors from center (j) to neighbors i and k
        v1 = coords_bohr[i] - coords_bohr[j]
        v2 = coords_bohr[k] - coords_bohr[j]
        if np.linalg.norm(v1) < 1e-8 or np.linalg.norm(v2) < 1e-8:
            continue
        # Unit normal to plane of v1 and v2
        n = np.cross(v1, v2)
        if np.linalg.norm(n) < 1e-8:
            # For linear or nearly linear angles
            arbitrary_axis = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(v1/np.linalg.norm(v1), arbitrary_axis)) > 0.9:
                arbitrary_axis = np.array([0.0, 1.0, 0.0])
            n = np.cross(v1, arbitrary_axis)
        n_unit = n / np.linalg.norm(n)
        # Compute displacements
        disp_i = 0.5 * np.cross(n_unit, v1)
        disp_k = 0.5 * np.cross(n_unit, v2)
        v = np.zeros((natm, 3))
        v[i] += disp_i
        v[k] += disp_k
        k_au = internal_force_constant(v)
        k_mdynA = k_au
        angle_k = k_mdynA
        atom_i = mol.GetAtomWithIdx(i)
        atom_j = mol.GetAtomWithIdx(j)
        atom_k = mol.GetAtomWithIdx(k)
        print(f"  Angle {atom_i.GetSymbol()}{i}-{atom_j.GetSymbol()}{j}-{atom_k.GetSymbol()}{k} force constant: {k_mdynA*1556.90013 } kj/A^2")

    return k_mdynA*1556.90013, bond_force*1556.90013, charges

def run_simulation(num_water=400, num_xenon=20, output_file='simulation.mp4'):
    """Run the molecular dynamics simulation with the specified parameters"""
    # -----------------------------
    # Simulation Box & Membrane
    # -----------------------------
    L = 40.0      # Box length in x,y,z (nm)
    membrane_x = L/2
    zn_spacing = 0.5
    zn_coords = []
    y_vals = np.arange(0, L+1e-9, zn_spacing)
    z_vals = np.arange(0, L+1e-9, zn_spacing)
    for yy in y_vals:
        for zz in z_vals:
            zn_coords.append([membrane_x, yy, zz])
    zn_coords = np.array(zn_coords)
    N_ZN = len(zn_coords)

    # -----------------------------
    # Water & Force Field
    # -----------------------------
    N_WATER = num_water
    N_XE = num_xenon

    dt = 0.001
    nsteps = 500
    save_interval = 1
    thermo_interval = 500
    use_thermostat = True
    temp_target = 300

    # For production use, we would run the const() function
    # However, for GitHub Actions we'll use pre-computed values
    # angle_k, bond_k, charges = const("O")
    
    # Use these pre-computed values instead to avoid running quantum calculations
    angle_k = 106.04721091  # kj/A^2
    bond_k = 484.34881621   # kj/A^2
    q_O, q_H = -0.834, 0.417  # charges

    # Harmonic bond & angle
    bond_length_eq = 0.09572
    angle_eq_deg = 104.52
    angle_eq = np.deg2rad(angle_eq_deg)

    # Masses
    mass_O = 15.9994
    mass_H = 1.008
    mass_Xe = 70
    mass_Zn = 65.38

    # -----------------------------
    # Allocate Arrays
    # -----------------------------
    N_mobile = N_WATER*3 + N_XE
    N = N_mobile + N_ZN

    positions = np.zeros((N, 3))
    velocities = np.zeros((N, 3))
    forces = np.zeros((N, 3))
    masses = np.zeros(N)
    charges = np.zeros(N)
    atom_type = np.zeros(N, dtype=int)

    # -----------------------------
    # Random Orientation Helper
    # -----------------------------
    def random_quaternion():
        u1, u2, u3 = np.random.rand(3)
        q = np.array([
            np.sqrt(1-u1)*np.sin(2*np.pi*u2),
            np.sqrt(1-u1)*np.cos(2*np.pi*u2),
            np.sqrt(u1)*np.sin(2*np.pi*u3),
            np.sqrt(u1)*np.cos(2*np.pi*u3)
        ])
        return q / np.linalg.norm(q)

    def quat_to_matrix(q):
        x, y, z, w = q
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
        ])

    # -----------------------------
    # 1) Init Water
    # -----------------------------
    placed_O = []
    for w in range(N_WATER):
        while True:
            xx = np.random.uniform(0.2, L/3)
            yy = np.random.uniform(0.2, L)
            zz = np.random.uniform(0.2, L)
            cand = np.array([xx, yy, zz])
            if all(np.linalg.norm(cand - o) > 0.25 for o in placed_O):
                placed_O.append(cand)
                break
        q = random_quaternion()
        rot = quat_to_matrix(q)
        iO = 3*w
        iH1 = iO+1
        iH2 = iO+2
        r0 = bond_length_eq
        # local coords
        H2x = r0*np.cos(angle_eq)
        H2y = r0*np.sin(angle_eq)
        Opos = cand
        H1pos = cand + rot.dot([r0, 0.0, 0.0])
        H2pos = cand + rot.dot([H2x, H2y, 0.0])
        positions[iO] = Opos
        positions[iH1] = H1pos
        positions[iH2] = H2pos
        atom_type[iO] = 0
        atom_type[iH1] = 1
        atom_type[iH2] = 1
        masses[iO] = mass_O
        masses[iH1] = mass_H
        masses[iH2] = mass_H
        charges[iO] = q_O
        charges[iH1] = q_H
        charges[iH2] = q_H

    # -----------------------------
    # 2) Init Xenon
    # -----------------------------
    start_xe = N_WATER*3
    placed_xe = []
    for i in range(N_XE):
        while True:
            xx = np.random.uniform(0.2, L/3)
            yy = np.random.uniform(0.2, L)
            zz = np.random.uniform(0.2, L)
            cand = np.array([xx, yy, zz])
            if all(np.linalg.norm(cand - p) > 0.3 for p in placed_xe):
                placed_xe.append(cand)
                break
        idx = start_xe + i
        positions[idx] = cand
        atom_type[idx] = 2
        masses[idx] = mass_Xe
        charges[idx] = 0  # Not using charges for xenon

    # -----------------------------
    # 3) Zn Membrane (fixed)
    # -----------------------------
    zn_start = N_mobile
    for i in range(N_ZN):
        positions[zn_start + i] = zn_coords[i]
        atom_type[zn_start + i] = 3
        masses[zn_start + i] = mass_Zn
        charges[zn_start + i] = 0.0

    # -----------------------------
    # 4) Velocities for Mobile
    # -----------------------------
    kB_kJ = 0.008314462618
    sigma_v = np.sqrt(kB_kJ*temp_target / masses[:N_mobile])
    velocities[:N_mobile] = np.random.normal(0., 1., (N_mobile, 3))*sigma_v[:, None]
    P = np.sum(masses[:N_mobile, None]*velocities[:N_mobile], axis=0)
    velocities[:N_mobile] -= P/np.sum(masses[:N_mobile])

    # -----------------------------
    # Force Calculation
    # -----------------------------
    def compute_forces(pos):
        """Harmonic O–H bonds, angle, ignoring LJ. Coulomb if charges set, up to cutoff=1.4."""
        F = np.zeros_like(pos)
        # Bond & angle
        for w in range(N_WATER):
            iO = 3*w
            iH1 = iO+1
            iH2 = iO+2
            # O–H1
            r1_vec = pos[iH1]-pos[iO]
            r1 = np.linalg.norm(r1_vec)
            if r1 < 1e-12:
                r1 = 1e-12
            dV1 = bond_k*(r1 - bond_length_eq)
            Fb1 = - dV1*(r1_vec/r1)
            F[iH1] += Fb1
            F[iO] -= Fb1
            # O–H2
            r2_vec = pos[iH2]-pos[iO]
            r2 = np.linalg.norm(r2_vec)
            if r2 < 1e-12:
                r2 = 1e-12
            dV2 = bond_k*(r2 - bond_length_eq)
            Fb2 = - dV2*(r2_vec/r2)
            F[iH2] += Fb2
            F[iO] -= Fb2
            # angle
            u1 = r1_vec/r1
            u2 = r2_vec/r2
            cosT = np.dot(u1, u2)
            cosT = np.clip(cosT, -1, 1)
            th = np.arccos(cosT)
            dth = th - angle_eq
            if abs(dth) > 1e-8:
                torque = angle_k*dth
                sinT = np.sqrt(1-cosT**2)
                if sinT > 1e-8:
                    e2p = u2 - cosT*u1
                    ne2p = np.linalg.norm(e2p)
                    if ne2p > 1e-8:
                        e2p /= ne2p
                        F1 = - torque/(r1*sinT) * e2p
                        F[iH1] += F1
                        F[iO] -= F1
                    e1p = u1 - cosT*u2
                    ne1p = np.linalg.norm(e1p)
                    if ne1p > 1e-8:
                        e1p /= ne1p
                        F2 = - torque/(r2*sinT) * e1p
                        F[iH2] += F2
                        F[iO] -= F2

        # Coulomb (optional) if charges set
        cutoff = 1.4
        coulomb_k = 138.9355
        for i in range(N-1):
            qi = charges[i]
            if abs(qi) < 1e-14:
                continue
            for j in range(i+1, N):
                qj = charges[j]
                if abs(qj) < 1e-14:
                    continue
                r_vec = pos[i]-pos[j]
                dist = np.linalg.norm(r_vec)
                if 0 < dist < cutoff:
                    inv_r2 = 1/(dist*dist)
                    Fmag = coulomb_k*qi*qj*inv_r2/dist
                    fij = Fmag*r_vec
                    F[i] += fij
                    F[j] -= fij
        return F

    # -----------------------------
    # Bond & Angle Clamping
    # -----------------------------
    def clamp_bonds_angles(pos):
        """
        After each step, forcibly rescale O–H distances above 0.12 nm down to near eq,
        and forcibly correct extreme angles if >30 deg from eq.
        """
        max_bond = 0.12   # if bond > 0.12 nm, rescale
        angle_tolerance_deg = 30.0
        angle_tolerance = np.deg2rad(angle_eq_deg + angle_tolerance_deg)
        angle_lower = np.deg2rad(angle_eq_deg - angle_tolerance_deg)
        for w in range(N_WATER):
            iO = 3*w
            iH1 = iO+1
            iH2 = iO+2
            # O–H1
            r1_vec = pos[iH1]-pos[iO]
            r1 = np.linalg.norm(r1_vec)
            if r1 > max_bond:
                # rescale to eq or something smaller
                scale = bond_length_eq/r1
                pos[iH1] = pos[iO] + scale*r1_vec
            # O–H2
            r2_vec = pos[iH2]-pos[iO]
            r2 = np.linalg.norm(r2_vec)
            if r2 > max_bond:
                scale = bond_length_eq/r2
                pos[iH2] = pos[iO] + scale*r2_vec
            # angle check
            u1 = (pos[iH1]-pos[iO])
            l1 = np.linalg.norm(u1)
            if l1 < 1e-12:
                continue
            u1 /= l1
            u2 = (pos[iH2]-pos[iO])
            l2 = np.linalg.norm(u2)
            if l2 < 1e-12:
                continue
            u2 /= l2
            cosT = np.dot(u1, u2)
            cosT = np.clip(cosT, -1, 1)
            th = np.arccos(cosT)
            # if angle is too large or too small, forcibly move H2
            if th > angle_tolerance or th < angle_lower:
                # we forcibly set angle to angle_eq
                # keep O–H1 in place, rotate H2 around O to fix angle
                axis = np.cross(u1, u2)
                norm_axis = np.linalg.norm(axis)
                if norm_axis < 1e-12:
                    continue
                axis /= norm_axis
                # desired delta = angle_eq - th
                dangle = angle_eq - th
                # rotate vector u2 around 'axis' by dangle
                # use Rodrigues' rotation formula
                u2_new = (u2*np.cos(dangle)
                          + np.cross(axis, u2)*np.sin(dangle)
                          + axis*np.dot(axis, u2)*(1-np.cos(dangle)))
                pos[iH2] = pos[iO] + l2*u2_new

    # -----------------------------
    # MD Integration
    # -----------------------------
    frames_dir = "frames"
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    frames_positions = []
    forces = compute_forces(positions)
    N_mobile = N_WATER*3 + N_XE

    print(f"Starting simulation with {N_WATER} water molecules and {N_XE} xenon atoms...")
    
    for step in range(nsteps):
        if step % 50 == 0:
            print(f"Step {step}/{nsteps}")
        # half-step velocity
        for i in range(N_mobile):
            velocities[i] += 0.5*dt*(forces[i]/masses[i])
        # position update
        oldpos = positions.copy()
        for i in range(N_mobile):
            positions[i] += velocities[i]*dt

        # box bounce
        for i in range(N_mobile):
            for d in range(3):
                if positions[i, d] < 0:
                    positions[i, d] *= -1
                    velocities[i, d] *= -1
                elif positions[i, d] > L:
                    positions[i, d] = 2*L - positions[i, d]
                    velocities[i, d] *= -1

        # clamp bond & angle
        clamp_bonds_angles(positions)

        # new forces
        forces = compute_forces(positions)
        # finalize velocity
        for i in range(N_mobile):
            velocities[i] += 0.5*dt*(forces[i]/masses[i])

        # thermostat
        if use_thermostat and (step+1) % thermo_interval == 0:
            KE = 0.5*np.sum(masses[:N_mobile]*np.sum(velocities[:N_mobile]**2, axis=1))
            dof = 3*N_mobile
            T_now = (2*KE)/(dof*kB_kJ)
            lam = np.sqrt(temp_target/T_now)
            velocities[:N_mobile] *= lam

        # store frames
        if step % save_interval == 0:
            frames_positions.append(positions.copy())

    print("Simulation done, saving frames...")

    # Indices
    O_idx = np.where(atom_type == 0)[0]
    H_idx = np.where(atom_type == 1)[0]
    Xe_idx = np.where(atom_type == 2)[0]
    Zn_idx = np.where(atom_type == 3)[0]

    for frame_idx, pos in enumerate(frames_positions):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.set_zlim(0, L)
        ax.set_xlabel("X (nm)")
        ax.set_ylabel("Y (nm)")
        ax.set_zlabel("Z (nm)")
        ax.set_title(f"Step {frame_idx*save_interval}")

        # plot O, H, Xe, Zn
        ax.scatter(pos[O_idx, 0], pos[O_idx, 1], pos[O_idx, 2], c='red', s=60, label='O')
        ax.scatter(pos[H_idx, 0], pos[H_idx, 1], pos[H_idx, 2], c='blue', s=20, label='H')
        ax.scatter(pos[Xe_idx, 0], pos[Xe_idx, 1], pos[Xe_idx, 2], c='purple', s=80, label='Xe')
        ax.scatter(pos[Zn_idx, 0], pos[Zn_idx, 1], pos[Zn_idx, 2], c='grey', s=40, alpha=0.6, label='Zn')

        # bond lines
        for w in range(N_WATER):
            iO = 3*w
            iH1 = iO+1
            iH2 = iO+2
            xdata = [pos[iO, 0], pos[iH1, 0]]
            ydata = [pos[iO, 1], pos[iH1, 1]]
            zdata = [pos[iO, 2], pos[iH1, 2]]
            ax.plot(xdata, ydata, zdata, 'k-', lw=1)
            xdata = [pos[iO, 0], pos[iH2, 0]]
            ydata = [pos[iO, 1], pos[iH2, 1]]
            zdata = [pos[iO, 2], pos[iH2, 2]]
            ax.plot(xdata, ydata, zdata, 'k-', lw=1)

        # Only add legend to the first frame to save time
        if frame_idx == 0:
            ax.legend()
        
        fname = os.path.join(frames_dir, f"frame_{frame_idx:04d}.png")
        plt.savefig(fname, dpi=100)
        plt.close(fig)

    print("All frames saved, creating video...")

    # Create video from frames
    try:
        from moviepy.editor import ImageSequenceClip
        
        image_files = sorted([os.path.join(frames_dir, img) for img in os.listdir(frames_dir) if img.endswith(".png")])
        clip = ImageSequenceClip(image_files, fps=24)
        clip.write_videofile(output_file, fps=24)
        
        # Clean up frames
        import shutil
        shutil.rmtree(frames_dir)
        
        print(f"Video saved as {output_file}")
    except ImportError:
        print("MoviePy not available. Frames saved in 'frames' directory.")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate Xenon Transport Simulation")
    parser.add_argument("--water", type=int, default=400, help="Number of water molecules")
    parser.add_argument("--xenon", type=int, default=20, help="Number of xenon atoms")
    parser.add_argument("--output", type=str, default="simulation.mp4", help="Output video file")
    parser.add_argument("--test", action="store_true", help="Run quantum chemistry test calculation")
    
    args = parser.parse_args()
    
    if args.test:
        try:
            # Test the quantum chemistry calculation with a small molecule
            print("Testing quantum chemistry calculation...")
            angle_k, bond_k, charges = const("O")
            print(f"Test successful! angle_k={angle_k}, bond_k={bond_k}")
        except Exception as e:
            print(f"Quantum chemistry test failed: {e}")
            sys.exit(1)
    else:
        # Run the simulation
        run_simulation(args.water, args.xenon, args.output)
