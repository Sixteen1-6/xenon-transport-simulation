import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from math import acos, degrees

# -----------------------------
# Simulation Parameters
# -----------------------------
def run_simulation(num_water=400, num_xenon=20, output_file='simulation.mp4'):
    """Run the molecular dynamics simulation with the specified parameters"""
    # -----------------------------
    # Simulation Box & Membrane
    # -----------------------------
    L = 40.0      # Box length in x,y,z (nm)
    membrane_x = L/2  # position of the membrane in the file
    zn_spacing = 0.5
    zn_coords = []
    y_vals = np.arange(0, L+1e-9, zn_spacing)
    z_vals = np.arange(0, L+1e-9, zn_spacing)
    for yy in y_vals:
        for zz in z_vals:
            zn_coords.append([membrane_x, yy, zz])
    zn_coords = np.array(zn_coords)
    N_ZN = len(zn_coords) # the number of zn atoms

    # -----------------------------
    # Water & Force Field
    # -----------------------------
    N_WATER = num_water
    N_XE = num_xenon

    dt = 0.001
    nsteps = 500
    save_interval = 5  # Save every 5 steps to reduce file size
    thermo_interval = 100
    use_thermostat = True
    temp_target = 300

    # Use these pre-computed values instead of running quantum calculations
    angle_k = 106.04721091  # kj/A^2
    bond_k = 484.34881621   # kj/A^2
    q_O, q_H = -0.834, 0.417  # charges

    # Harmonic bond & angle
    bond_length_eq = 0.09572
    angle_eq_deg = 104.52
    angle_eq = np.deg2rad(angle_eq_deg)

    # Masses Atomic in amu
    mass_O = 15.9994
    mass_H = 1.008
    mass_Xe = 70
    mass_Zn = 65.38

    # -----------------------------
    # Allocate Arrays
    # -----------------------------
    N_mobile = N_WATER*3 + N_XE
    N = N_mobile + N_ZN # total number of particles

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

    def quat_to_matrix(q): # converts a unit quartien to a 3X3 rotaion matrix in x,y,z,w order
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
        # Local geometry for water: H1 at [r0, 0, 0], H2 at [r0*cos(angle), r0*sin(angle), 0]
        H2x = r0*np.cos(angle_eq)
        H2y = r0*np.sin(angle_eq)
        Opos = cand
        H1pos = cand + rot.dot([r0, 0.0, 0.0])         # Rotate local coords to random orientation, the
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
    
    # Reduce number of steps based on system size for performance
    if N_WATER > 500 or N_XE > 50:
        nsteps = min(nsteps, 300)  # Fewer steps for large systems
    
    for step in range(nsteps):
        if step % 50 == 0:
            print(f"Step {step}/{nsteps}")
        # half-step velocity
        for i in range(N_mobile):
            velocities[i] += 0.5*dt*(forces[i]/masses[i])
        # position update
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

        # store frames (less frequently to reduce file size)
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

        # plot O, H, Xe, Zn - use smaller marker sizes for larger systems
        size_factor = max(0.2, min(1.0, 200 / (N_WATER + N_XE)))
        
        ax.scatter(pos[O_idx, 0], pos[O_idx, 1], pos[O_idx, 2], c='red', s=60*size_factor, label='O')
        ax.scatter(pos[H_idx, 0], pos[H_idx, 1], pos[H_idx, 2], c='blue', s=20*size_factor, label='H')
        ax.scatter(pos[Xe_idx, 0], pos[Xe_idx, 1], pos[Xe_idx, 2], c='purple', s=80*size_factor, label='Xe')
        
        # For large systems, only plot a subset of the zinc atoms to improve performance
        if len(Zn_idx) > 1000:
            sample_size = min(1000, len(Zn_idx) // 4)
            sample_indices = np.random.choice(Zn_idx, sample_size, replace=False)
            ax.scatter(pos[sample_indices, 0], pos[sample_indices, 1], pos[sample_indices, 2], 
                      c='grey', s=40*size_factor, alpha=0.6, label='Zn')
        else:
            ax.scatter(pos[Zn_idx, 0], pos[Zn_idx, 1], pos[Zn_idx, 2], 
                      c='grey', s=40*size_factor, alpha=0.6, label='Zn')

        # For large systems, only draw bonds for a subset of water molecules
        max_bonds = 200
        if N_WATER <= max_bonds:
            water_range = range(N_WATER)
        else:
            # Sample water molecules near the membrane for bond drawing
            water_indices = []
            for w in range(N_WATER):
                iO = 3*w
                if positions[iO, 0] > L/3 and positions[iO, 0] < 2*L/3:
                    water_indices.append(w)
            if len(water_indices) > max_bonds:
                water_indices = np.random.choice(water_indices, max_bonds, replace=False)
            water_range = water_indices
            
        # Draw bonds for selected water molecules
        for w in water_range:
            iO = 3*w
            iH1 = iO+1
            iH2 = iO+2
            xdata = [pos[iO, 0], pos[iH1, 0]]
            ydata = [pos[iO, 1], pos[iH1, 1]]
            zdata = [pos[iO, 2], pos[iH1, 2]]
            ax.plot(xdata, ydata, zdata, 'k-', lw=0.5)
            xdata = [pos[iO, 0], pos[iH2, 0]]
            ydata = [pos[iO, 1], pos[iH2, 1]]
            zdata = [pos[iO, 2], pos[iH2, 2]]
            ax.plot(xdata, ydata, zdata, 'k-', lw=0.5)

        # Only add legend to the first frame
        if frame_idx == 0:
            ax.legend()
        
        # Remove grid and tick labels to reduce visual clutter
        ax.grid(False)
        if frame_idx > 0:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
        
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
    except Exception as e:
        print(f"Error creating video: {e}")
        print("Frames saved in 'frames' directory.")
        
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate Xenon Transport Simulation")
    parser.add_argument("--water", type=int, default=400, help="Number of water molecules")
    parser.add_argument("--xenon", type=int, default=20, help="Number of xenon atoms")
    parser.add_argument("--output", type=str, default="simulation.mp4", help="Output video file")
    
    args = parser.parse_args()
    
    # Run the simulation
    run_simulation(args.water, args.xenon, args.output)
