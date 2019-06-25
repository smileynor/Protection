import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def calculateZone1_normal(imp_of_line):
    return imp_of_line*0.85

def calculateZone2_normal(imp_of_line):
    return imp_of_line*1.2

def calculateZone3_normal(imp_of_line,imp_of_next_line):
    return (imp_of_line+imp_of_next_line)*1.05

def calculateZone1_with_margin(imp_of_line,margin_as_percent):
    return imp_of_line*(1-margin_as_percent/100)

def calculateZone2_with_margin(imp_of_line,margin_as_percent):
    return imp_of_line*(1+margin_as_percent/100)

def calculateZone3_with_margin(imp_of_line,imp_of_next_line,margin_as_percent):
    return (imp_of_line+imp_of_next_line)*(1+margin_as_percent/100)

def findHighestReactanceOfInfeed(r,x):
    #Chech every point up intill 135 deg is reached.
    return "Not implemented yet"

def findLongestDistanceOfInfeed(r,x):
    return "Not implemented yet"

# Settings: z1, r1, a1, b1, g1
class siprotec_7ST63x_settings(object):
    """Settings for Siprotec 7ST63x
    Values are given and saved as primary values
    Parameters
    ----------
     - *z1* : Impedance reach of first zone
     - *z2* : Impedance reach of second zone
     - *z3* : Impedance reach of third zone
     - *r1* : Resistance reach of first zone
     - *r2* : Resistance reach of second zone
     - *r3* : Resistance reach of third zone
     - *a1* : Alpha reach of first zone (first quadrant)
     - *a2* : Alpha reach of second zone (first quadrant)
     - *a3* : Alpha reach of third zone (first quadrant)
     - *b1* : Beta reach of first zone (2. quadrant)
     - *b2* : Beta reach of second zone (2. quadrant)
     - *b3* : Beta reach of third zone (2. quadrant)
     - *g1* : Gamma reach of first zone (4. quadrant, defined positive)
     - *g2* : Gamma reach of second zone (4. quadrant, defined positive)
     - *g3* : Gamma reach of third zone (4. quadrant, defined positive)
    """
    paths = []
    CT_ratio_P1 = 800
    CT_ratio_P2 = 5
    VT_ratio_P1 = 16000
    VT_ratio_P2 = 110

    def __init__(self, z1, z2, z3, r1=5.0, r2=5.0, r3=5.0, a1=30, a2=30, a3=30, b1=135, b2=135, b3=135, g1=15, g2=15, g3=15, dx=0, dy=0):
        self.z1 = np.float(z1)
        self.z2 = np.float(z2)
        self.z3 = np.float(z3)
        self.r1 = np.float(r1)
        self.r2 = np.float(r2)
        self.r3 = np.float(r3)
        self.a1 = np.float(a1)
        self.a2 = np.float(a2)
        self.a3 = np.float(a3)
        self.b1 = np.float(b1)
        self.b2 = np.float(b2)
        self.b3 = np.float(b3)
        self.g1 = np.float(g1)
        self.g2 = np.float(g2)
        self.g3 = np.float(g3)
        self.dx = np.float(dx)
        self.dy = np.float(dy)

    def improvement(self):
        # Draw_zone1(2,3) kan lages generisk, slik at fra draw_all så hentes
        # draw_Zone1(ax,**kwargs), der **kwargs == facecolor='red', alpha=0.4, label="Sone 1"
        # draw_Zone1(ax,**kwargs), der **kwargs == facecolor='blue', alpha=0.3, label="Sone 2"
        # draw_Zone1(ax,**kwargs), der **kwargs == facecolor='green', alpha=0.3, label="Sone 3"
        # og **kwargs legges til ved patch i hver av dem.
        # i stedet for å hente self.verdier så kan de sendes inn til en generell.
        pass


    def draw_zone1(self, ax):
        """Code for drawing Zone 1
        Parameter
        ---------
        *ax* : matplotlib.axes where the drawing is to be added"""
        points = []
        n = np.linspace(0,1,30)

        # First point
        points.append(tuple([np.float(0),np.float(0)]))
        # Points in 4. quadrant
        points.append(tuple([self.r1,-self.r1*np.sin(np.deg2rad(self.g1))]))
        # Points in 1. quadrant
        points.append(tuple([self.r1, self.r1 * np.sin(np.deg2rad(self.a1))]))
        points.append(tuple([self.z1 * np.cos(np.deg2rad(self.a1)), self.z1 * np.sin(np.deg2rad(self.a1))]))
        # Cirkle line towards 2. quadrant
        x1 = self.z1 * np.cos(np.deg2rad(self.a1 + n * (self.b1 - self.a1)))
        y1 = self.z1 * np.sin(np.deg2rad(self.a1 + n * (self.b1 - self.a1)))

        # Drawing procedure
        codes = [matplotlib.path.Path.MOVETO,
                 matplotlib.path.Path.LINETO,
                 matplotlib.path.Path.LINETO,
                 matplotlib.path.Path.LINETO]

        for x, y in np.nditer([x1, y1]):
            points.append(tuple([np.float(x), np.float(y)]))
            codes.append(matplotlib.path.Path.LINETO)

        points.append(tuple([np.float(0), np.float(0)]))
        codes.append(matplotlib.path.Path.CLOSEPOLY)

        # Movement?
        if self.dy != 0 or self.dx != 0:
            new_position = []
            for i in points:
                new_position.append(tuple([list(i)[0] + self.dx, list(i)[1] + self.dy]))

            points = new_position

        # Edges to vertices (points)
        path = matplotlib.path.Path(points, codes)
        patch = matplotlib.patches.PathPatch(path, facecolor='red', alpha=0.4, label="Sone 1")

        # Draw zone
        ax.add_patch(patch)

        # Save path
        self.paths.append(path)
        return path

    def draw_zone2(self, ax):
        """Code for drawing Zone 2
        Parameter
        ---------
        *ax* : matplotlib.axes where the drawing is to be added"""
        points = []
        n = np.linspace(0, 1, 30)

        # First point
        points.append(tuple([np.float(0), np.float(0)]))
        # Points in 4. quadrant
        points.append(tuple([self.r2, -self.r2 * np.sin(np.deg2rad(self.g2))]))
        # Points in 1. quadrant
        points.append(tuple([self.r2, self.r2 * np.sin(np.deg2rad(self.a2))]))
        points.append(tuple([self.z2 * np.cos(np.deg2rad(self.a2)), self.z2 * np.sin(np.deg2rad(self.a2))]))
        # Cirkle line towards 2. quadrant
        x2 = self.z2 * np.cos(np.deg2rad(self.a2 + n * (self.b2 - self.a2)))
        y2 = self.z2 * np.sin(np.deg2rad(self.a2 + n * (self.b2 - self.a2)))

        # Drawing procedure
        codes = [matplotlib.path.Path.MOVETO,
                 matplotlib.path.Path.LINETO,
                 matplotlib.path.Path.LINETO,
                 matplotlib.path.Path.LINETO]

        for x, y in np.nditer([x2, y2]):
            points.append(tuple([np.float(x), np.float(y)]))
            codes.append(matplotlib.path.Path.LINETO)

        points.append(tuple([np.float(0), np.float(0)]))
        codes.append(matplotlib.path.Path.CLOSEPOLY)

        # Movement?
        if self.dy != 0 or self.dx != 0:
            new_position = []
            for i in points:
                new_position.append(tuple([list(i)[0] + self.dx, list(i)[1] + self.dy]))

            points = new_position

        # Edges to vertices (points)
        path = matplotlib.path.Path(points, codes)
        patch = matplotlib.patches.PathPatch(path, facecolor='blue', alpha=0.3, label="Sone 2")

        # Draw zone
        ax.add_patch(patch)

        # Save path
        self.paths.append(path)
        return path

    def draw_zone3(self, ax):
        """Code for drawing Zone 3
        Parameter
        ---------
        *ax* : matplotlib.axes where the drawing is to be added"""
        points = []
        n = np.linspace(0, 1, 30)

        # First point
        points.append(tuple([np.float(0), np.float(0)]))
        # Points in 4. quadrant
        points.append(tuple([self.r3, -self.r3 * np.sin(np.deg2rad(self.g3))]))
        # Points in 1. quadrant
        points.append(tuple([self.r3, self.r3 * np.sin(np.deg2rad(self.a3))]))
        points.append(tuple([self.z3 * np.cos(np.deg2rad(self.a3)), self.z3 * np.sin(np.deg2rad(self.a3))]))
        # Cirkle line towards 2. quadrant
        x3 = self.z3 * np.cos(np.deg2rad(self.a3 + n * (self.b3 - self.a3)))
        y3 = self.z3 * np.sin(np.deg2rad(self.a3 + n * (self.b3 - self.a3)))

        # Drawing procedure
        codes = [matplotlib.path.Path.MOVETO,
                 matplotlib.path.Path.LINETO,
                 matplotlib.path.Path.LINETO,
                 matplotlib.path.Path.LINETO]

        for x, y in np.nditer([x3, y3]):
            points.append(tuple([np.float(x), np.float(y)]))
            codes.append(matplotlib.path.Path.LINETO)

        points.append(tuple([np.float(0), np.float(0)]))
        codes.append(matplotlib.path.Path.CLOSEPOLY)

        # Movement?
        if self.dy != 0 or self.dx != 0:
            new_position = []
            for i in points:
                new_position.append(tuple([list(i)[0] + self.dx, list(i)[1] + self.dy]))

            points = new_position

        # Edges to vertices (points)
        path = matplotlib.path.Path(points, codes)
        patch = matplotlib.patches.PathPatch(path, facecolor='green', alpha=0.3, label="Sone 3")

        # Draw zone
        ax.add_patch(patch)

        # Save path
        self.paths.append(path)
        return path

    def scale_ax(self,ax):
        """Function for scaling the diagram according to zone 3
        Parameter
        ---------
        *ax* : matplotlib.axes where the drawings are to be added
        """
        limit = np.max(np.ceil(np.absolute(self.z3)))

        ax.set_xlim(min(np.cos(np.deg2rad(self.b1))*self.z1,
                        np.cos(np.deg2rad(self.b2))*self.z2,
                        np.cos(np.deg2rad(self.b3))*self.z3,
                        -2.0) * 1.2, limit * 1.2)
        ax.set_ylim(min(np.tan(np.deg2rad(self.b1))*self.r1,
                        np.tan(np.deg2rad(self.b1))*self.r2,
                        np.tan(np.deg2rad(self.b1))*self.r3,
                        -2.0) * 1.2, limit * 1.2)

        # drawing of guiding lines
        for i in range(1, int(self.z3 / 5) + 2):
            circles = matplotlib.patches.Circle((0, 0), radius=5 * i, linestyle=':', fill=False, alpha=0.3)
            ax.add_patch(circles)

    def draw_all(self, ax):
        """ Draws the diagram in the order:
        Z3, Z2, Z1, grid lines"""
        self.draw_zone3(ax)
        self.draw_zone2(ax)
        self.draw_zone1(ax)
        self.scale_ax(ax)

    def to_secondary_values(self):
        """ Print impedances in secondary values
        Primary Z = U / I

        Secondary current I' = I * CT_secondary/CT_primary
        Secondary voltage U' = U * VT_secondary/VT_primary

        Secondary Z' = U' / I' = U * VT_secondary/VT_primary / (I * CT_secondary/CT_primary)
            Z' = Z * (VT_secondary/VT_primary)/(CT_secondary/CT_primary)

        """
        scale = (self.VT_ratio_P2/self.VT_ratio_P1)/(self.CT_ratio_P2/self.CT_ratio_P1)
        for i in [self.z1, self.z2, self.z3]:
            print("Prim: {0} ohm ->(*{1}) = {2} ohm (Secondary)".format(i,np.round(scale,2),np.round(i*scale,2)))

    def change_CT_ratio(self,CT_prim, CT_sek):
        """Function for changing the current transformer ratio.
        Default: 800/5 A"""
        self.CT_ratio_P1 = CT_prim
        self.CT_ratio_P2 = CT_sek

    def change_VT_ratio(self,VT_prim, VT_sek):
        """Function for changing the voltage transformer ratio.
        Default: 16000/110 V"""
        self.VT_ratio_P1 = VT_prim
        self.VT_ratio_P2 = VT_sek


class ABB_REO517_settings(object):
    """Settings for Siprotec 7ST63x
        Values are given and saved as primary values
        Parameters
        ----------
         - *r1*: Resistance reach of first zone
         - *r2*: Resistance reach of second zone
         - *r3*: Resistance reach of third zone
         - *x1*: Reactance reach of first zone
         - *x2*: Reactance reach of second zone
         - *x3*: Reactance reach of third zone
         - *a1*: Alpha reach of first zone (first quadrant)
         - *a2*: Alpha reach of second zone (first quadrant)
         - *a3*: Alpha reach of third zone (first quadrant)
         - *g1*: Gamma reach of first zone (4. quadrant, defined positive)
         - *g2*: Gamma reach of second zone (4. quadrant, defined positive)
         - *g3*: Gamma reach of third zone (4. quadrant, defined positive)
        """
    paths = []
    CT_ratio_P1 = 800
    CT_ratio_P2 = 5
    VT_ratio_P1 = 16000
    VT_ratio_P2 = 110

    def __init__(self, r1, r2, r3, x1, x2, x3, a=30, g=15, dx=0, dy=0):
        self.r1 = np.float(r1)
        self.r2 = np.float(r2)
        self.r3 = np.float(r3)
        self.x1 = np.float(x1)
        self.x2 = np.float(x2)
        self.x3 = np.float(x3)
        self.a = np.float(a)
        self.g = np.float(g)
        self.dx = np.float(dx)
        self.dy = np.float(dy)

    def draw_zone1(self, ax, **kwargs):
        return(self.draw_zone(ax,H1=self.x1, R1=self.r1, facecolor='red', alpha=0.3, label="Sone 1", **kwargs))

    def draw_zone2(self, ax, **kwargs):
        return (self.draw_zone(ax, H1=self.x2, R1=0, R2=self.r2, facecolor='blue', alpha=0.3, label="Sone 2", **kwargs))

    def draw_zone3(self, ax, **kwargs):
        return (self.draw_zone(ax, H1=self.x3, R1=0, R2=self.r3, facecolor='green', alpha=0.3, label="Sone 3", **kwargs))

    def draw_zone(self, ax, H1 = 5, R1 = 5, R2 = None, **kwargs):
        """Code for drawing Zone 1

        Parameter
        ---------
        *ax* : matplotlib.axes where the drawing is to be added
            F_____E__________________ / D
             \    |                 /
              \   |           ... /G
               \  |    . ../J    /
                \ |A_/____H____C__________
                    \    /
                        \I
                            \B

            :param H1: Height of X
            :param R1: Resistance blender
            :param R2: Resistance length
            :param Gamma: Angle in degree CAB
            :param Alpha: Angle in degree CAG
            :param Beta: Angle in degree CAF
            :param Theta: Angle in degree ACB

        Return
        ---------
        ax.path
        """

        # getting the settings:
        Gamma = self.g
        Alpha = self.a
        Beta = 115  # 2. quadrant
        Theta = 12  # degree from vertical line

        if R2 is None:
            R2 = R1

        nTan = lambda x: np.tan(np.deg2rad(x))
        # n = lambda x: np.deg2rad(x)

        A = [0, 0]
        B = [(nTan(90 - Theta) * R2) / (nTan(Gamma) + nTan(90 - Theta)),
             -(nTan(Gamma) * nTan(90 - Theta) * R2) / (nTan(Gamma) + nTan(90 - Theta))]
        # (TanC*R2)/(tanA+tanC)
        # AND -(tanA*tanC*R2)/(tanA+tanC)
        C = [R2, 0]
        D = [R2 + H1 * nTan(Theta), H1]  # R2+H1*tan(ACB) AND H1
        E = [0, H1]
        F = [-nTan(Beta - 90) * H1, H1]  # -(tan(Beta-90)*H1) AND H1
        G = [-nTan(90 - Theta) * R2 / (nTan(Alpha) - nTan(90 - Theta)),
             nTan(Alpha) * (-nTan(90 - Theta) * R2 / (nTan(Alpha) - nTan(90 - Theta)))]
        # R2 + (tanA*tan(180-ACB)*R2)/(tanA+tan(180-ACB))/Tan(CAG)
        # AND (tanA*tan(180-ACB)*R2)/(tanA+tan(180-ACB))
        H = [R1, 0]
        I = [(nTan(90 - Theta) * R1) / (nTan(Gamma) + nTan(90 - Theta)),
             -(nTan(Gamma) * nTan(90 - Theta) * R1) / (nTan(Gamma) + nTan(90 - Theta))]
        # (TanC*R1)/(tanA+tanC)
        # AND -(tanA*tanC*R1)/(tanA+tanC)
        J = [-nTan(90 - Theta) * R1 / (nTan(Alpha) - nTan(90 - Theta)),
             nTan(Alpha) * (-nTan(90 - Theta) * R1 / (nTan(Alpha) - nTan(90 - Theta)))]
        # R1 + (tanA*tan(180-ACB)*R1)/(tanA+tan(180-ACB))/Tan(CAG)
        # AND (tanA*tan(180-ACB)*R1)/(tanA+tan(180-ACB))

        points = []
        codes = [matplotlib.path.Path.MOVETO]

        # keyPoints = [A,B,C,D,E,F,G,H,I,J]
        keyPoints = [A, I, H, J, G, D, E, F]

        for point in range(len(keyPoints) - 1):
            codes.append(matplotlib.path.Path.LINETO)

        for x, y in keyPoints:
            points.append(tuple([x, y]))

        points.append(tuple([np.float(0), np.float(0)]))
        codes.append(matplotlib.path.Path.CLOSEPOLY)

        # Movement?
        if self.dy != 0 or self.dx != 0:
            new_position = []
            for i in points:
                new_position.append(tuple([list(i)[0] + self.dx, list(i)[1] + self.dy]))

            points = new_position

        # Updating axis
        height_upper_limit = H1 * 1.2
        height_lower_limit = B[1] * 1.5
        vertical_upper_limit = D[0] * 1.2
        vertical_lower_limit = F[0] * 1.2

        if ax.get_ylim()[0] < height_lower_limit:
            height_lower_limit = ax.get_ylim()[0]

        if ax.get_ylim()[1] > height_upper_limit:
            height_upper_limit = ax.get_ylim()[1]

        if ax.get_xlim()[0] < vertical_lower_limit:
            vertical_lower_limit = ax.get_xlim()[0]

        if ax.get_xlim()[1] > vertical_upper_limit:
            vertical_upper_limit = ax.get_xlim()[1]

        ax.set_xlim(vertical_lower_limit, vertical_upper_limit)
        ax.set_ylim(height_lower_limit, height_upper_limit)

        ax.axhline(y=0, color='gray', linestyle=':')
        ax.axvline(x=0, color='gray', linestyle=':')
        ax.set_xlabel(xlabel="R [$\Omega$]")
        ax.set_ylabel(ylabel="X [$\Omega$]")
        ax.set_title("Tittel")

        # Edges to vertices (points)
        path = matplotlib.path.Path(points, codes)
        patch = matplotlib.patches.PathPatch(path, **kwargs)

        # Draw zone
        ax.add_patch(patch)

        # Save path
        self.paths.append(path)
        return path

    def draw_all(self, ax):
        """ Draws the diagram in the order:
        Z3, Z2, Z1, grid lines"""
        self.draw_zone3(ax)
        self.draw_zone2(ax)
        self.draw_zone1(ax)
        #self.scale_ax(ax)


class RZYBE_settings(object):
    """Settings for ASEA RYZBE
        Values are given and saved as primary values
        Parameters
        ----------
         - *R_kl*: Resistance of contact line
         - *X_kl*: Reaktance of contact line
         - *P1*: Setting parameter 1
         - *P2*: Setting parameter 2
        """
    paths = []
    CT_ratio_P1 = 800
    CT_ratio_P2 = 5
    VT_ratio_P1 = 16000
    VT_ratio_P2 = 110

    # Settings
    Zk = 3.95 # Internal impedance of protection relay
    a_ratio = 1 # Scaling factor
    Zr = 0 # Length of distance from Origo to end point along protection line


    def __init__(self, R_kl, X_kl, phi_1=72, phi_2=14, phi_k=47, margin_1=0.85, margin_2=1.2):
        self.R_kl = R_kl
        self.X_kl = X_kl
        self.phi_1 = phi_1
        self.phi_2 = phi_2
        self.phi_k = phi_k
        self.margin_1 = margin_1
        self.margin_2 = margin_2

    def draw_zone1(self, ax):
        pass

    def draw_zone2(self, ax):
        pass

    def draw_all(self, ax):
        pass

    def to_secondary_values(self):
        pass

    def change_CT_ratio(self,CT_prim, CT_sek):
        """Function for changing the current transformer ratio.
        Default: 800/5 A"""
        self.CT_ratio_P1 = CT_prim
        self.CT_ratio_P2 = CT_sek

    def change_VT_ratio(self,VT_prim, VT_sek):
        """Function for changing the voltage transformer ratio.
        Default: 16000/110 V"""
        self.VT_ratio_P1 = VT_prim
        self.VT_ratio_P2 = VT_sek

    def how_to_calculate_P1(self):
        print("P1 = cos(phi_1 - phi_KL) / cos(phi_1 - phi_K) * (a*100*l_k)/(L_1)")

    def how_to_calculate_P2(self):
        print("P2 = cos(phi_2 - phi_KL) / cos(phi_2 - phi_K) * (a*100*l_k)/(L_2)")

    def find_Zr_from_KL(R_kl, X_kl, Zr_angle):
        Z_kl = np.complex(R_kl, X_kl)
        Zr_angle_of_length_1 = np.complex(np.cos(np.deg2rad(Zr_angle)), np.sin(np.deg2rad(Zr_angle)))
        Zr = np.real(Z_kl * np.conj(Z_kl)) / np.real(Zr_angle_of_length_1 * np.conj(Z_kl))
        return Zr

    def example(self):
        """
        def find_Zr_from_KL(R_kl, X_kl,Zr_angle):
            Z_kl=np.complex(R_kl,X_kl)
            Zr_angle_of_length_1 = np.complex(np.cos(np.deg2rad(Zr_angle)),np.sin(np.deg2rad(Zr_angle)))
            Zr = np.real(Z_kl*np.conj(Z_kl))/np.real(Zr_angle_of_length_1*np.conj(Z_kl))
            return Zr

        degrees = np.linspace(0,360,361)
        #angle_of_circleline = 72 # degree
        zone1_angle = 72 # degree
        zone2_angle = 14 # degree
        angle_constant = 47 # degree
        R_kl = 0.200*73
        X_kl = 0.190*73

        Zr = find_Zr_from_KL(R_kl, X_kl, zone1_angle)
        x_mid_point_circleline = 0.5*Zr*np.cos(np.deg2rad(zone1_angle))
        y_mid_point_circleline = 0.5*Zr*np.sin(np.deg2rad(zone1_angle))
        length_to_midpoint = np.sqrt(x_mid_point_circleline**2+y_mid_point_circleline**2)
        xs = x_mid_point_circleline + length_to_midpoint*np.cos(np.deg2rad(degrees))
        ys = y_mid_point_circleline + length_to_midpoint*np.sin(np.deg2rad(degrees))

        Zr2 = find_Zr_from_KL(R_kl, X_kl, zone2_angle)
        x_mid_point_circleline2 = 0.5*Zr2*np.cos(np.deg2rad(zone2_angle))
        y_mid_point_circleline2 = 0.5*Zr2*np.sin(np.deg2rad(zone2_angle))
        length_to_midpoint2 = np.sqrt(x_mid_point_circleline2**2+y_mid_point_circleline2**2)
        xs2 = x_mid_point_circleline2 + length_to_midpoint2*np.cos(np.deg2rad(degrees))
        ys2 = y_mid_point_circleline2 + length_to_midpoint2*np.sin(np.deg2rad(degrees))

        fig, ax = plt.subplots(1)
        ax.plot(xs,ys)
        ax.plot(xs2,ys2)
        ax.plot([0,R_kl],[0,X_kl])
        ax.hlines(0,2*length_to_midpoint,-1*length_to_midpoint)
        ax.vlines(0,2*length_to_midpoint,-1*length_to_midpoint)


        plt.show()
        :return:
        """

class impedanceDiagram(object):
    """Class for impedance Diagram

    Example:
    ---------
    import Protection.distance as prot
    a = prot.impedanceDiagram("U1")
    a.test()

    Example 2:
    ----------
    a = prot.impedanceDiagram("U1")
    a.add_first_section(4.0,4.0,"A")
    a.add_second_section(3.0,3.0,"A","B")
    a.add_third_section(4.0,4.0,"B","C")
    a.add_third_section(5.0,3.0,"B","D")
    a.draw_setup()
    a.draw_first_section()
    a.draw_second_section()
    a.draw_third_section()
    b = prot.siprotec_7ST63x_settings(5,8,13)
    b.draw_all(a.ax)

    a.draw_load_at_beginning()

    for r,x,i in a.first_sections:
        a.draw_load_at_end(r,x)
        a.draw_min_SSC_at_beginning(4)
        a.draw_min_SSC_at_section_end(4,r,x)
        a.draw_infeed(4,4,b.paths[1])

    plt.show()

    """
    ax = ""
    fig = ""
    settings = []
    first_sections = []
    second_sections = []
    third_sections = []
    overcurrent_setting = 800 # A
    undervoltage_setting = 10 # kV
    max_load_angle = 60 # degree


    def __init__(self, name):
        self.name = name

    def draw_setup(self):
        """figure, ax, xlim, ylim ..."""
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        #self.ax.set_xlim(-40, 40) # MÅ LAGES MER FLEKSIBEL
        #self.ax.set_ylim(-10, 50) # MÅ LAGES MER FLEKSIBEL
        self.ax.axhline(y=0, color='gray', linestyle=':')
        self.ax.axvline(x=0, color='gray', linestyle=':')
        self.ax.set_xlabel(xlabel="R [$\Omega$]")
        self.ax.set_ylabel(ylabel="X [$\Omega$]")
        self.ax.set_title(self.name)

    def draw_ending(self):
        """ Finishing up the drawing by adding legends, etx"""
        self.ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()

    def draw_first_section(self):
        """ Function for drawing sections towards next stations.
        r,x,name at end"""
        for r,x,name_at_end in self.first_sections:
            self.ax.plot(r, x, 'o', color='gray')
            self.ax.annotate(name_at_end, (r, x), horizontalalignment='right', xytext=(0, 2), textcoords='offset points')
            # ax.quiver(0,0,rlinje,xlinje,color="gray", scale_units='xy', angles='xy', scale=1, zorder=10, alpha=1)
            self.ax.plot([0, r], [0, x], color="k")

    def draw_second_section(self):
        """ Functioin for drawing sections behind the next station
        r,x,name_at_start, name_at_end
        """
        for r,x,name_at_start,name_at_end in self.second_sections:
            for r0, x0, name_at_end0 in self.first_sections:
                if name_at_start == name_at_end0:
                    self.ax.plot(r0 + r, x0 + x, 'o', color='gray')
                    self.ax.annotate(name_at_end, (r0 + r, x0 + x), horizontalalignment='left',
                        xytext=(0, 2), textcoords='offset points')
                    # ax.quiver(rlinje1,xlinje1,rlinje2,xlinje2,color="gray", scale_units='xy', angles='xy', scale=1, zorder=10, alpha=1)
                    self.ax.plot([r0, r0 + r], [x0, x0 + x], color="k")

    def draw_third_section(self):
        """ Functioin for drawing sections behind the second station
            r,x,name_at_start, name_at_end
            """
        for r,x,name_at_start,name_at_end in self.third_sections:
            for r1, x1, name_at_start1, name_at_end1 in self.second_sections:
                for r0, x0, name_at_end0 in self.first_sections:
                    if name_at_start == name_at_end1:
                        if name_at_start1 == name_at_end0:
                            self.ax.plot(r0+r1 + r, x0+x1 + x, 'o', color='gray')
                            self.ax.annotate(name_at_end, (r0+r1 + r, x0+x1 + x), horizontalalignment='left',
                                xytext=(0, 2), textcoords='offset points')
                            # ax.quiver(rlinje1,xlinje1,rlinje2,xlinje2,color="gray", scale_units='xy', angles='xy', scale=1, zorder=10, alpha=1)
                            self.ax.plot([r0+r1, r0+r1 + r], [x0+x1, x0+x1 + x], color="k")


    def draw_load_at_beginning(self):
        """ Function for drawing load at beginning of section
        Draws points at 7 intervall between 0 and maximum phi.
        If 60 degree, one pr 10 degree.

        Method:
            Rp = |U|^2 / (S_load*cos(phi))
            Xp = |U|^2 / (S_load*sin(phi))
            q = Rp/Xp
            Rs = Rp/(q^2+1)
            Xs = Rs*q

        Typical values S = 10 MVA
        U = 10 kV
        phi = 0 - 60 deg
        """
        U = self.undervoltage_setting
        S = self.undervoltage_setting*self.overcurrent_setting/1000
        phi_max = self.max_load_angle

        # Gjør det mulig å dele på 0
        np.seterr(divide="ignore")

        ax = self.ax
        n = 7
        Rp = np.abs(U) * np.abs(U) / (S * np.cos(np.deg2rad(np.linspace(0.001, phi_max, n))))
        Xp = np.abs(U) * np.abs(U) / (S * np.sin(np.deg2rad(np.linspace(0.001, phi_max, n))))
        q = np.divide(Rp, Xp)

        Rs = np.divide(Rp, q * q + 1)
        Xs = Rs * q

        ax.plot(Rs, Xs, ".", color="k", label="Last start")
        ax.annotate("Last start {mw:d} MW \n {kv:d} kV 0-60\xb0 ".format(mw=int(S), kv=int(U)),
                    (Rs[-1], Xs[-1]),
                    size=7,
                    horizontalalignment='right',
                    xytext=(2, 2),
                    textcoords='offset points')

    def draw_load_at_end(self,r1,x1):
        """ Function for drawing load at end of section
        Draws points at 7 intervall between 0 and maximum phi.
        If 60 degree, one pr 10 degree.

        Method:
            Rp = |U|^2 / (S_load*cos(phi))
            Xp = |U|^2 / (S_load*sin(phi))
            q = Rp/Xp
            Rs = Rp/(q^2+1) + section_resistance
            Xs = Rs*q + sectioin_reactance

        Typical values S = 10 MVA
        U = 10 kV
        phi = 0 - 60 deg

        """
        np.seterr(divide="ignore")

        ax = self.ax
        U = self.undervoltage_setting
        S = self.undervoltage_setting*self.overcurrent_setting/1000  # (kV * A / 1000 ) = MVA
        phi_max = self.max_load_angle
        rlinje = r1
        xlinje = x1

        n = 7
        Rp = np.abs(U) * np.abs(U) / (S * np.cos(np.deg2rad(np.linspace(0.001, phi_max, n))))
        Xp = np.abs(U) * np.abs(U) / (S * np.sin(np.deg2rad(np.linspace(0.001, phi_max, n))))
        q = np.divide(Rp, Xp)

        Rs = np.divide(Rp, q * q + 1)
        Xs = Rs * q

        ax.plot(Rs + rlinje, Xs + xlinje, ".", color="gray", label="Last slutt")
        ax.annotate("Last slutt {mw:d} MW \n {kv:d} kV 0-60\xb0 ".format(mw=int(S), kv=int(U)),
                    (rlinje + Rs[0], xlinje + Xs[0]),
                    size=7,
                    horizontalalignment='left',
                    xytext=(0, -5),
                    textcoords='offset points')

    def draw_infeed(self,r1,x1,path):
        """ Function for drawing the infeed curve

        Formula:
            Z_seen_from_relay = U^2 / ((U^2/Z) - P_infeed)
            *U*: lowest accepted voltage
            *Z*: impedance of section
            *P_infeed*: infeed power from train or branch.

        """
        n = 101 # Number of points
        rlinje = r1
        xlinje = x1
        U = self.undervoltage_setting  # kV, lowest accepted voltage of protection
        ax = self.ax
        P = np.linspace(0, 100, n)  # MW, power infeed from train
        Z = np.array([rlinje + 1j * xlinje])
        Z_t = 0

        Z_t_inside = []
        highP1 = 0

        been_Outside = False

        # Finner alle P som er dekket av vernet
        U2 = U * U
        # Cheching each points of infeed power
        for p in P:
            if not been_Outside:
                # Performing the calculation
                Z_tilbake = np.divide(U2, (U2 / Z) - p)

                #Checking if the apparent impedance is inside of protection area
                if path.contains_point([Z_tilbake.real, Z_tilbake.imag]):
                    # If inside, add point and update highest infeed power
                    Z_t_inside.append(Z_tilbake)
                    highP1 = p
                else:
                    been_Outside = True

        # For more accuracy:
        P = np.linspace(highP1, highP1 + 1, 11)
        antallPunkt = len(Z_t_inside)
        for p in P:
            Z_tilbake = np.divide(U2, (U2 / Z) - p)
            if path.contains_point([Z_tilbake.real, Z_tilbake.imag]):
                highP1 = p
                if len(Z_t_inside) > antallPunkt:
                    Z_t_inside[antallPunkt] = Z_tilbake
                else:
                    Z_t_inside.append(Z_tilbake)



        Z_t_inside = np.array(Z_t_inside)

        lastr = Z_t_inside[-1].real
        lastx = Z_t_inside[-1].imag

        ax.plot(Z_t_inside.real, Z_t_inside.imag, ".", color="r", label="Tilbakemating")
        ax.annotate("$P_{tilbake}$=%.1f MW" % highP1, (lastr, lastx), horizontalalignment='right', xytext=(0, 2),
                    textcoords='offset points')

    def find_highest_reactance_of_infeed(self,r1,x1):
        """ Function for finding the highest reactance based on infeed

                Formula:
                    Z_seen_from_relay = U^2 / ((U^2/Z) - P_infeed)
                    *U*: lowest accepted voltage
                    *Z*: impedance of section
                    *P_infeed*: infeed power from train or branch.

                """
        n = 500  # Number of points
        rlinje = r1
        xlinje = x1
        U = self.undervoltage_setting  # kV, lowest accepted voltage of protection
        ax = self.ax
        P = np.linspace(0, 150, n)  # MW, power infeed from train
        Z = np.array([rlinje + 1j * xlinje])
        highP1 = 0

        # Finner alle P som er dekket av vernet
        U2 = U * U
        # Cheching each points of infeed power
        for p in P:
            # Performing the calculation
            Z_tilbake = np.divide(U2, (U2 / Z) - p)

            # Checking if the apparent impedance is higher than earlier
            if Z_tilbake.imag[0] > highP1:
                # If higher update highest infeed power
                highP1 = Z_tilbake.imag[0]

        return highP1


    def draw_min_SSC_at_beginning(self,r_min):
        self.ax.plot([r_min],[0],"o",color="b")
        self.ax.annotate("Min. kortsl",
                    (r_min,0),
                    size=7,
                    horizontalalignment='right',
                    xytext=(2, 2),
                    textcoords='offset points')

    def draw_min_SSC_at_section_end(self, r_min, r1,x1):
        self.ax.plot([r_min+r1], [x1], "o", color="b")
        self.ax.annotate("Min. kortsl",
                         (r_min+r1, x1),
                         size=7,
                         horizontalalignment='right',
                         xytext=(2, 2),
                         textcoords='offset points')

    def add_first_section(self,r,x,name_at_end):
        self.first_sections.append([r,x,name_at_end])

    def add_second_section(self,r,x,name_at_start,name_at_end):
        self.second_sections.append([r, x, name_at_start, name_at_end])

    def add_third_section(self,r,x,name_at_start,name_at_end):
        self.third_sections.append([r, x, name_at_start, name_at_end])

    def set_overcurrent_value(self,current):
        self.overcurrent_setting=current

    def set_undervoltage_value(self,voltage):
        self.undervoltage_setting=voltage

    def test(self):
        #self.add_first_section(5.0,3.0,"Oslo")
        self.add_first_section(3.0, 8.0, "Moss")
        self.add_second_section(3.0, 5.0, "Oslo", "U")
        self.add_second_section(5.0, 5.0, "Oslo", "I")
        self.add_second_section(5.0, 8.0, "Moss", "J")
        self.add_third_section(2.0,2.0,"I","O")
        self.add_third_section(1.0, 5.0, "I", "P")
        self.add_third_section(2.0, 2.0, "J", "A")
        self.add_third_section(1.0, 5.0, "J", "B")
        self.draw_setup()
        self.draw_first_section()
        self.draw_second_section()
        self.draw_third_section()

        #Draw last
        self.draw_load_at_beginning()
        for r,x,i in self.first_sections:
            self.draw_load_at_end(r,x)
            self.draw_min_SSC_at_beginning(4)
            self.draw_min_SSC_at_section_end(4,r,x)



def network_data():
    return "Not implementet yet"

def todo():
    print("Implementer MHO-karakteristikk")
    print("Implementer beregninger for innstillinger")
    print("Implementer ABB-vern med polygonal-zoner")
    print("Implementer autosettings")
    print("Implementer netverk som pandas-dataframe")
    print("Implementer legg til impedansediagram fra neste stasjon over")
    print("HTML-rapport med settinger")
    print("HTML-rapport med metode")
    print("HTML-rapport med alerts")
    print("max motstand i SSC")

def polytest(H1, R1, R2=None, Gamma=15, Alpha=30, Beta=135, Theta=12, **kwargs):
    """
    F_____E__________________ / D
     \    |                 /
      \   |           ... /G
       \  |    . ../J    /
        \ |A_/____H____C__________
            \    /
                \I
                    \B

    :param H1: Height of X
    :param R1: Resistance blender
    :param R2: Resistance length
    :param Gamma: Angle in degree CAB
    :param Alpha: Angle in degree CAG
    :param Beta: Angle in degree CAF
    :param Theta: Angle in degree ACB
    :return: drawing
    """
    if R2 is None:
        R2 = R1

    nTan = lambda x: np.tan(np.deg2rad(x))
    n = lambda x: np.deg2rad(x)

    A = [0, 0]
    B = [(nTan(90-Theta)*R2)/(nTan(Gamma)+nTan(90-Theta)),
         -(nTan(Gamma)*nTan(90-Theta)*R2)/(nTan(Gamma)+nTan(90-Theta))]
        # (TanC*R2)/(tanA+tanC)
        # AND -(tanA*tanC*R2)/(tanA+tanC)
    C = [R2, 0]
    D = [R2+H1*nTan(Theta), H1] # R2+H1*tan(ACB) AND H1
    E = [0, H1]
    F = [-nTan(Beta-90)*H1, H1] # -(tan(Beta-90)*H1) AND H1
    G = [-nTan(90-Theta)*R2/(nTan(Alpha)-nTan(90-Theta)),
        nTan(Alpha)*(-nTan(90-Theta)*R2/(nTan(Alpha)-nTan(90-Theta)))]
        # R2 + (tanA*tan(180-ACB)*R2)/(tanA+tan(180-ACB))/Tan(CAG)
        # AND (tanA*tan(180-ACB)*R2)/(tanA+tan(180-ACB))
    H = [R1, 0]
    I = [(nTan(90-Theta)*R1)/(nTan(Gamma)+nTan(90-Theta)),
         -(nTan(Gamma)*nTan(90-Theta)*R1)/(nTan(Gamma)+nTan(90-Theta))]
        # (TanC*R1)/(tanA+tanC)
        # AND -(tanA*tanC*R1)/(tanA+tanC)
    J = [-nTan(90-Theta)*R1/(nTan(Alpha)-nTan(90-Theta)),
        nTan(Alpha)*(-nTan(90-Theta)*R1/(nTan(Alpha)-nTan(90-Theta)))]
        # R1 + (tanA*tan(180-ACB)*R1)/(tanA+tan(180-ACB))/Tan(CAG)
        # AND (tanA*tan(180-ACB)*R1)/(tanA+tan(180-ACB))

    points = []
    codes=[matplotlib.path.Path.MOVETO]

    #keyPoints = [A,B,C,D,E,F,G,H,I,J]
    keyPoints = [A, I, H, J, G, D, E, F]

    for point in range(len(keyPoints)-1):
        codes.append(matplotlib.path.Path.LINETO)


    for x, y in keyPoints:
        points.append(tuple([x,y]))

    points.append(tuple([np.float(0), np.float(0)]))
    codes.append(matplotlib.path.Path.CLOSEPOLY)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Updating axis
    height_upper_limit = H1 * 1.2
    height_lower_limit = B[1] * 1.5
    vertical_upper_limit = D[0] * 1.2
    vertical_lower_limit = F[0] * 1.2

    if ax.get_ylim()[0] < height_lower_limit:
        height_lower_limit = ax.get_ylim()[0]

    if ax.get_ylim()[1] > height_upper_limit:
        height_upper_limit = ax.get_ylim()[1]

    if ax.get_xlim()[0] < vertical_lower_limit:
        vertical_lower_limit = ax.get_xlim()[0]

    if ax.get_xlim()[1] > vertical_upper_limit:
        vertical_upper_limit = ax.get_xlim()[1]

    ax.set_xlim(vertical_lower_limit, vertical_upper_limit)
    ax.set_ylim(height_lower_limit, height_upper_limit)

    ax.axhline(y=0, color='gray', linestyle=':')
    ax.axvline(x=0, color='gray', linestyle=':')
    ax.set_xlabel(xlabel="R [$\Omega$]")
    ax.set_ylabel(ylabel="X [$\Omega$]")
    ax.set_title("Tittel")

    # Edges to vertices (points)
    path = matplotlib.path.Path(points, codes)
    patch = matplotlib.patches.PathPatch(path, facecolor='red', alpha=0.3, label="Sone1", **kwargs)

    # Draw zone
    ax.add_patch(patch)

    plt.show()

#polytest(10,3,5)

# Example
"""
a = impedanceDiagram("U1")
a.add_first_section(4.0,4.0,"A")
a.add_second_section(3.0,3.0,"A","B")
a.add_third_section(4.0,4.0,"B","C")
a.add_third_section(5.0,3.0,"B","D")
a.draw_setup()
a.draw_first_section()
a.draw_second_section()
a.draw_third_section()
b = ABB_REO517_settings(r1=4, r2=5, r3=6, x1=10, x2=11, x3=12, a=30, g=15, dx=0, dy=0)
b.draw_all(a.ax)

a.draw_load_at_beginning()

for r,x,i in a.first_sections:
    a.draw_load_at_end(r,x)
    a.draw_min_SSC_at_beginning(4)
    a.draw_min_SSC_at_section_end(4,r,x)
    a.draw_infeed(4,4,b.paths[1])

plt.show()

"""
