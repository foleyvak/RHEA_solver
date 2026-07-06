#!/usr/bin/env python3
# Plot the C-D nozzle simulation results from an HDF5 snapshot (nozzle_cd_<iter>.h5).
# Usage: python3 plot_nozzle.py [file.h5]   (default: latest nozzle_cd_*.h5)
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt, h5py, glob, sys

fn = sys.argv[1] if len(sys.argv)>1 else sorted(glob.glob("nozzle_cd_*.h5"))[-1]
f = h5py.File(fn, "r")
X = np.array(f["x"])*1e3; Y = np.array(f["y"])*1e3   # mm, shape (gNx,gNy)
Ma= np.array(f["Ma"]);    P = np.array(f["P"])/1e5   # bar
rho=np.array(f["rho"]);   u = np.array(f["u"])
it = int(f.attrs["Iteration"]); t = float(f.attrs["Time"])
gNx,gNy = Ma.shape
jc = 0  # centreline (South symmetry axis, first interior row)
xc = X[:,jc]; radius = Y[:,gNy-1]   # wall = last interior row

# ---- centreline ----
fig,ax=plt.subplots(3,1,figsize=(9,9),sharex=True)
ax[0].plot(xc,radius,'k-',lw=2); ax[0].fill_between(xc,radius,radius.max()*1.05,color='0.85')
ax[0].set_ylabel("wall radius [mm]"); ax[0].set_title(f"C-D nozzle (curvilinear C++ port), iter {it}, t={t:.3e}s : centreline")
ax[1].plot(xc,Ma[:,jc],'b-',lw=2); ax[1].axhline(1.0,color='r',ls='--',lw=1,label="Ma=1"); ax[1].set_ylabel("Mach"); ax[1].legend()
ax[2].plot(xc,P[:,jc],'g-',lw=2); ax[2].set_ylabel("P [bar]"); ax[2].set_xlabel("x [mm]")
it_th=int(np.argmin(radius))
for a in ax: a.grid(alpha=0.3); a.axvline(xc[it_th],color='0.5',ls=':',lw=1)
plt.tight_layout(); plt.savefig("nozzle_centerline.png",dpi=120); plt.close()
print("nozzle_centerline.png : throat Ma=%.3f exit Ma=%.3f exit P=%.2f bar"%(Ma[it_th,jc],Ma[-1,jc],P[-1,jc]))

# ---- 2D Mach field (mirrored about axis) ----
fig,ax=plt.subplots(figsize=(11,5))
c=ax.contourf(X, Y, Ma, levels=40, cmap="jet"); ax.contourf(X,-Y, Ma, levels=40, cmap="jet")
plt.colorbar(c,ax=ax,label="Mach")
ax.plot(X[:,-1], Y[:,-1],'k-',lw=1.5); ax.plot(X[:,-1],-Y[:,-1],'k-',lw=1.5)
ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_title(f"Nozzle Mach field (mirrored), iter {it}")
plt.tight_layout(); plt.savefig("nozzle_Ma_field.png",dpi=120); plt.close()
print("nozzle_field.png written from", fn)

# ---- 2D Velocity field (mirrored about axis) ----
fig,ax=plt.subplots(figsize=(11,5))
c=ax.contourf(X, Y, u, levels=40, cmap="jet"); ax.contourf(X,-Y, u, levels=40, cmap="jet")
plt.colorbar(c,ax=ax,label="Velocity")
ax.plot(X[:,-1], Y[:,-1],'k-',lw=1.5); ax.plot(X[:,-1],-Y[:,-1],'k-',lw=1.5)
ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_title(f"Nozzle Velocity field (mirrored), iter {it}")
plt.tight_layout(); plt.savefig("nozzle_u_field.png",dpi=120); plt.close()
print("nozzle_u_field.png written from", fn)

# ---- 2D density field (mirrored about axis) ----
fig,ax=plt.subplots(figsize=(11,5))
c=ax.contourf(X, Y, rho, levels=40, cmap="jet"); ax.contourf(X,-Y, rho, levels=40, cmap="jet")
plt.colorbar(c,ax=ax,label="Density")
ax.plot(X[:,-1], Y[:,-1],'k-',lw=1.5); ax.plot(X[:,-1],-Y[:,-1],'k-',lw=1.5)
ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_title(f"Nozzle Density field (mirrored), iter {it}")
plt.tight_layout(); plt.savefig("nozzle_rho_field.png",dpi=120); plt.close()
print("nozzle_rho_field.png written from", fn)
