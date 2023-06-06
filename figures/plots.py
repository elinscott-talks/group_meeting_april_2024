import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.tri import Triangulation


def bare_figure(two_d=False, labelnmu=False, nmu_height=0):
    if two_d:
        _, ax = plt.subplots()
    else:
        ax = plt.axes(projection='3d')
    f = plt.gcf()
    f.set_figheight(4.8)
    f.set_figwidth(5)
    if two_d:
        ax.set_ylim([0, u/4])
    else:
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_zlim([0, u/4])
        ax.set_xlabel(r'$n^\uparrow$')
        ax.set_ylabel(r'$n^\downarrow$')
        plt.subplots_adjust(0, 0.03, 1, 1)
        ax.view_init(azim=20, elev=30)

        if labelnmu:
            ax.plot([0.5, 0.7], [1.3, 1.5], [nmu_height, nmu_height], 'k', clip_on=False)
            ax.text(0.75, 1.55, nmu_height, r'$n$', clip_on=False)
            ax.plot([0.7, 0.5], [1.3, 1.5], [nmu_height, nmu_height], 'k', clip_on=False)
            ax.text(0.5, 1.55, nmu_height, r'$\mu$', clip_on=False)

    return ax


def j_correction(J, nup, ndown, minority=True):
    correction = nup*ndown
    if minority:
        correction -= np.minimum(nup, ndown)
    return J*correction


def u_correction(U, nup, ndown):
    return U/2*(nup*(1 - nup) + ndown*(1 - ndown))


def novel_u(Uup, Udown, nup, ndown):
    ntilde = nup + ndown - 1
    return Uup/2*(np.abs(ntilde) + (1 - 2*nup)*ntilde) + Udown/2*(np.abs(ntilde) + (1 - 2*ndown)*ntilde)


def novel_u_potential(Uup, Udown, nup, ndown, sigma=0):
    ntilde = nup + ndown - 1
    safe_ntilde = copy.deepcopy(ntilde)
    safe_ntilde[safe_ntilde == 0] == np.nan
    potential = np.zeros(nup.shape)
    for i, (u, n) in enumerate(((Uup, nup), (Udown, ndown))):
        potential += u/4*(np.abs(ntilde)/safe_ntilde + 1 - 2*n)
        if i == sigma:
            potential -= ntilde*u/2
    potential[np.isnan(potential)] = 0
    return potential


def novel_k(K, nup, ndown):
    ntilde = nup + ndown - 1
    return K/4*((nup - ndown)**2 - (np.abs(ntilde) - 1)**2)


def novel_k_potential(K, nup, ndown, sigma=0):
    ntilde = nup + ndown - 1
    safe_ntilde = copy.deepcopy(ntilde)
    safe_ntilde[safe_ntilde == 0] == np.nan
    potential = np.zeros(nup.shape)
    if sigma == 0:
        potential += nup - ndown
    else:
        potential += ndown - nup
    potential += np.abs(ntilde)/safe_ntilde - ntilde
    potential[np.isnan(potential)] = 0
    potential *= K/2
    return potential


if __name__ == '__main__':

    rc = {"font.family": "serif",
          "mathtext.fontset": "cm"}
    plt.rcParams.update(rc)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

    # Setup
    npoints = 41
    u = 1
    ones = np.ones(npoints)
    x = np.linspace(0, 1, npoints)
    X, Y = np.meshgrid(x, x)
    n = np.linspace(0, 2, 2*npoints - 1)
    mu = np.linspace(-1, 1, 2*npoints - 1)

    def plot_trisurf(ax, Z, **kwargs):
        # Assemble triangles
        triangles = []
        for i in range(npoints):
            for j in range(npoints):
                if i != npoints - 1 and j != npoints - 1:
                    triangles.append([npoints * i + j, npoints*i + j + 1, npoints*(i + 1) + j])
                if i != 0 and j != 0:
                    triangles.append([npoints * i + j, npoints*i + j - 1, npoints*(i - 1) + j])
        X, Y = np.meshgrid(x, x)
        triang = Triangulation(X.flatten(), Y.flatten(), triangles)
        # zissou = ["#3b9ab2", "#78b7c5", "#ebcc2a", "#e1af00", "#f43915"]
        # zissoucm = ListedColormap(sns.blend_palette(zissou, 256))
        ax.azim = 20
        ax.set_xlabel(r'$n^\uparrow$', fontsize=20, rotation=0, labelpad=10)
        ax.set_ylabel(r'$n^\downarrow$', fontsize=20, rotation=0, labelpad=10)
        plt.subplots_adjust(top=1.05, left=0.05, right=1, bottom=0.05)
        return ax.plot_trisurf(X.flatten(), Y.flatten(), triangles=triangles,
                               Z=Z.flatten(), linewidth=0, antialiased=False, edgecolor='none', **kwargs)

    # The +U correction
    ax = bare_figure(labelnmu=False)
    surf = ax.plot_surface(X, Y, u_correction(u, X, Y), rstride=1, cstride=1, cmap='inferno', edgecolor='none')
    plt.savefig('u_correction.pdf', format='pdf')
    plt.close()
#
    # Add paths
    ax = bare_figure(labelnmu=False)
    surf = ax.plot_surface(X, Y, u_correction(u, X, Y), rstride=1, cstride=1,
                           cmap='inferno', edgecolor='none', alpha=0.5)
    surf.set_alpha(0.5)
    ax.plot3D(x, 0.8*ones, u_correction(u, x, 0.8*ones), 'r')
    ax.plot3D(0.9*ones, x, u_correction(u, 0.9*ones, x), 'g')
    ax.plot3D(x, x, u_correction(u, x, x), 'b')
    ax.plot3D(x, 1-x, u_correction(u, x, 1-x), 'k')
    plt.savefig('u_correction_with_paths.pdf', format='pdf')
    plt.close()
#
    # The +J correction
    ax = bare_figure(labelnmu=False, nmu_height=-0.25)
    j = 1
    surf = ax.plot_surface(X, Y, j_correction(j, X, Y), rstride=1, cstride=1, cmap='inferno', edgecolor='none')
    ax.set_zlim([-0.25, 0.05])
    plt.savefig('j_correction.pdf', format='pdf')
    plt.close()
#
    # # The +U+J correction
    # ax = bare_figure()
    # j = 0.2
    # surf = ax.plot_surface(X, Y, u_correction(u, X, Y) + j_correction(j, X, Y), rstride=1, cstride=1, cmap='inferno', edgecolor='none')
    # plt.savefig('u_j_correction.pdf', format='pdf')
#
    # ax = bare_figure(True)
    # nup = 1/2*(1 + mu)
    # ndown = 1/2*(1 - mu)
    # ax.plot(mu, u_correction(u, nup, ndown) + j_correction(j, nup, ndown))
    # ax.set_xlabel('$\mu$')
    # ax.set_xlim([-1, 1])
    # plt.tight_layout()
    # plt.savefig('u_j_correction_2d.pdf', format='pdf')
#
    # # The +Ueff+J correction
    # ax = bare_figure()
    # surf = ax.plot_surface(X, Y, u_correction(u-j, X, Y) + j_correction(j, X, Y), rstride=1, cstride=1, cmap='inferno', edgecolor='none')
    # plt.savefig('ueff_j_correction.pdf', format='pdf')
#
    # ax = bare_figure(True)
    # ax.plot(mu, u_correction(u - j, nup, ndown) + j_correction(j, nup, ndown))
    # ax.set_xlabel('$\mu$')
    # ax.set_xlim([-1, 1])
    # plt.tight_layout()
    # plt.savefig('ueff_j_correction_2d.pdf', format='pdf')
#
    # # The +U+Jno minority correction
    # ax = bare_figure()
    # surf = ax.plot_surface(X, Y, u_correction(u, X, Y) + j_correction(j, X, Y, False), rstride=1, cstride=1, cmap='inferno', edgecolor='none')
    # plt.savefig('u_jnomin_correction.pdf', format='pdf')
#
    # ax = bare_figure(True)
    # ax.plot(n, u_correction(u, n/2, n/2) + j_correction(j, n/2, n/2, False))
    # ax.set_xlabel('$n$')
    # ax.set_xlim([0, 2])
    # ax.set_ylim([0, 0.35])
    # plt.tight_layout()
    # plt.savefig('u_jnomin_correction_2d.pdf', format='pdf')
#
    # # The +Ueff+Jno minority correction
    # ax = bare_figure()
    # surf = ax.plot_surface(X, Y, u_correction(u - j, X, Y) + j_correction(j, X, Y, False), rstride=1, cstride=1, cmap='inferno', edgecolor='none')
    # plt.savefig('ueff_jnomin_correction.pdf', format='pdf')
#
    # ax = bare_figure(True)
    # ax.plot(n, u_correction(u - j, n/2, n/2) + j_correction(j, n/2, n/2, False))
    # ax.set_xlabel('$n$')
    # ax.set_xlim([0, 2])
    # ax.set_ylim([0, 0.35])
    # plt.tight_layout()
    # plt.savefig('ueff_jnomin_correction_2d.pdf', format='pdf')

    # Novel U functional
    ax = bare_figure()
    # surf = ax.plot_surface(X, Y, novel_u(1, 1, X, Y), rstride=1, cstride=1, cmap='inferno', edgecolor='none')
    surf = plot_trisurf(ax, novel_u(1, 1, X, Y), cmap='inferno')
    plt.savefig('msie_correction.pdf', format='pdf')
    plt.close()

    # Novel U functional
    ax = bare_figure(labelnmu=False)
    surf = plot_trisurf(ax, novel_u(0.5, 1, X, Y), cmap='inferno')
    # surf = ax.plot_surface(X, Y, novel_u(0.5, 1, X, Y), rstride=1, cstride=1, cmap='inferno', edgecolor='none')
    plt.savefig('msie_plus_asym_correction.pdf', format='pdf')
    plt.close()

    # Novel U potential
    ax = bare_figure()
    surf = ax.plot_surface(X, Y, novel_u_potential(0.5, 1, X, Y), rstride=1,
                           cstride=1, cmap='inferno', edgecolor='none')
    ax.set_zlim([-0.6, 0.6])
    plt.savefig('novel_u_potential.pdf', format='pdf')
    plt.close()

    # In 2d
    fig, ax = plt.subplots(1, 1)
    z = novel_u_potential(0.5, 1, X, Y)
    dx = x[1] - x[0]
    cf = ax.pcolormesh(X, Y, z, shading='gouraud', cmap='coolwarm')
    fig.colorbar(cf, ax=ax)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel(r'$n^\uparrow$')
    ax.set_ylabel(r'$n^\downarrow$')
    plt.savefig('novel_u_potential_2d.pdf', format='pdf')
    plt.close()

    # Novel K functional
    ax = bare_figure()
    # surf = ax.plot_surface(X, Y, novel_k(1, X, Y), rstride=1, cstride=1, cmap='inferno', edgecolor='none')
    surf = plot_trisurf(ax, novel_k(1, X, Y), cmap='inferno')
    ax.set_zlim([-0.25, 0])
    plt.savefig('sce_correction.pdf', format='pdf')
    plt.close()

    # Novel K potential
    ax = bare_figure()
    surf = ax.plot_surface(X, Y, novel_k_potential(1, X, Y), rstride=1, cstride=1, cmap='inferno', edgecolor='none')
    ax.set_zlim([-1, 1])
    plt.savefig('novel_k_potential.pdf', format='pdf')
    plt.close()

    # In 2d
    fig, ax = plt.subplots(1, 1)
    z = novel_k_potential(1, X, Y)
    dx = x[1] - x[0]
    cf = ax.pcolormesh(X, Y, z, shading='gouraud', cmap='coolwarm')
    fig.colorbar(cf, ax=ax)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel(r'$n^\uparrow$')
    ax.set_ylabel(r'$n^\downarrow$')
    plt.savefig('novel_k_potential_2d.pdf', format='pdf')
    plt.close('all')
