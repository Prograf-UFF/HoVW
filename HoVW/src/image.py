import cv2
import operator
import numpy as np
from tree import ImageTree
from descriptors import ZernikeMoments as ZM, GeometricDescriptors as GD

class Mask:
    """The mask represents a shape of the image.

    Parameters
    ----------
    cnt: Shape's contour.
    outline: Shape's outline.

    Attributes
    ----------
    contour: array, shape = [list of points, point, point's coordinates]
        Shape's contour edges points.
    center_radius: tuple = (x, y, radius)
        Geometric center (point) of the shape and the radius between
        this point and the farthest edge point. 
    centroid_radius: tuple = (x, y, radius)
        Center of mass (point) of the shape and the radius between
        this point and the farthest edge point.
    outline: array, shape = [height, width]
        Shape's outline.
    area: float
        Shape's area.
    perimeter: float
        Shape's perimeter.
    feature_vector: array, shape = [features]
        Features which describes the shape.
    """

    def __init__(self, cnt, outline):
        self.contour = cnt
        self.outline = outline
        self.center_radius = self._center_radius()
        self.centroid_radius = self._centroid_radius()
        self.area = cv2.contourArea(self.contour)
        self.perimeter = cv2.arcLength(self.contour,True)
        self.feature_vector = self._build_feature_vector()

    def _center_radius(self):
        center,radius = cv2.minEnclosingCircle(self.contour)

        return (center[0], center[1],radius)

    def _centroid_radius(self):
        moment = cv2.moments(self.contour)
        cx = int(moment['m10']/moment['m00'])
        cy = int(moment['m01']/moment['m00'])

        ext = []
        ext.append(tuple(self.contour[self.contour[:, :, 0].argmin()][0])) #left
        ext.append(tuple(self.contour[self.contour[:, :, 0].argmax()][0])) #Right
        ext.append(tuple(self.contour[self.contour[:, :, 1].argmin()][0])) #Top
        ext.append(tuple(self.contour[self.contour[:, :, 1].argmax()][0])) #Bottom
        r = -1
        for e in ext:
            x = np.sqrt((cx-e[0])**2 + (cy-e[1])**2)
            r = x if x>r else r

        return (cx, cy, r)

    def _build_feature_vector(self):
        """Build the feature vector.
            
            Format:
            convexity, compactness, eccentricity, avg_bending_energy, ZMs

        Returns
        -------
        array, shape = [features]
        Array with the features.
        """

        convexity = np.array([GD.convexity(self.contour, self.perimeter)])
        compactness = np.array([GD.compactness(self.area, self.perimeter)])
        eccentricity = np.array([GD.eccentricity(self.contour, 
            (self.centroid_radius[0], self.centroid_radius[1]))])
        avg_bending_energy = np.array(
            [GD.avg_bending_energy(self.contour, 1)])
        
        geo_vec = np.append(convexity, [compactness, eccentricity, 
            avg_bending_energy])

        desc = ZM()
        zernike_moments = desc.describe(self.outline, self.center_radius[2])
        
        return np.append(geo_vec, zernike_moments)

    def show(self, overload=False):
        """Show shape.

        Parameters
        ----------
        overload: Show circles relating to center and center of mass.
        """

        header = 'Mask'
        img = self.outline

        if overload:
            header = 'Mask Complete'
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            cv2.circle(img, (int(self.centroid_radius[0]),int(self.centroid_radius[1])),\
             int(self.centroid_radius[2]), (0,0,255))
            cv2.circle(img, (int(self.centroid_radius[0]),int(self.centroid_radius[1])),\
             2, (0, 0, 255), -1)

            cv2.circle(img, (int(self.center_radius[0]),int(self.center_radius[1])),\
             int(self.center_radius[2]), (0,255,0))
            cv2.circle(img, (int(self.center_radius[0]),int(self.center_radius[1])),\
             2, (0,255,0), -1)

        cv2.imshow(header, img)
        cv2.waitKey(0)

    def draw(self, output=None, header=None, overload=False):
        """Draw the shape.

        Parameters
        ----------
        overload: Show circles relating to center and center of mass.
        output: Path where the image should be drawn.
        header: Final image's name.
        """

        final_header = 'mask'
        img = self.outline

        if overload:
            final_header = 'mask-complete'
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            cv2.circle(img, (int(self.centroid_radius[0]),int(self.centroid_radius[1])), \
            2, (0, 0, 255), -1)
            cv2.circle(img, (int(self.centroid_radius[0]),int(self.centroid_radius[1])), \
            int(self.centroid_radius[2]), (0, 0, 255))

            cv2.circle(img, (int(self.center_radius[0]),int(self.center_radius[1])), \
            2, (0, 255, 0), -1)
            cv2.circle(img, (int(self.center_radius[0]),int(self.center_radius[1])), \
            int(self.center_radius[2]), (0, 255, 0))


        if header: final_header = header
        if output: final_header = output + final_header

        final_header += '.png'
        cv2.imwrite(final_header, img)

class Image:
    """Image representation.

    Parameters
    ----------
    path: Path where the image is located.

    Attributes
    ----------
    path: string
        Original image's path.
    original: array, shape = [height, width, channels]
        Image itself.
    grayscale: array, shape = [height, width]
        Image grayscale version.
    threshold: array, shape = [height, width]
        Image binary threshold.
    tree: ImageTree
        Hierarchical (tree) representation of the image,
        segmented by its shapes.
    """

    def __init__(self, path):
        self.path = path
        self.original = cv2.imread(path)
        self.grayscale = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        self.threshold = self._get_threshold()
        self.tree = self._set_tree()

    def _get_threshold(self):
        blur = cv2.medianBlur(self.grayscale, 5)
        blur = cv2.bilateralFilter(blur,9,75,75)
        _,threshold = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #THRESH_BINARY = fundo preto or THRESH_BINARY_INV = fundo branco
        
        return threshold

    def _get_hierarchy_masks(self):

        outline = np.zeros(self.grayscale.shape, dtype = "uint8")

        (_, cnts, hierarchy) = cv2.findContours(self.threshold.copy(),
            cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        masks = []
        i = 0
        hierarchy = hierarchy[0]
        alter_hie = []
        for cnt in cnts:
            if(cv2.contourArea(cnt) > 0):
                masks.append([cnt, cv2.drawContours(outline.copy(),
                 [cnt], -1, 255, -1)]) #contour, outline
                alter_hie.append(np.append(hierarchy[i], 1))
            else:
                alter_hie.append(np.append(hierarchy[i], 0))
            i += 1
        hierarchy = np.array([alter_hie])

        return (hierarchy[0], masks)

    def _set_tree(self):
        (hierarchy, shapes) = self._get_hierarchy_masks()
        masks = []
        for mask in shapes:
            masks.append(Mask(mask[0],mask[1]))
        
        tree = ImageTree(hierarchy, masks, self.path)

        # 1 = acima da metade; #0 = toda a imagem
        tree.cut_off(0.2)

        return tree
    
    def draw(self, output=None):
        """Draw the image by its shapes.
        
        Parameters
        ----------
        output: Path where the image should be drawn.
        """
        masks = self.tree.get_tree_masks()
        for m,i in zip(masks, range(len(masks))):
            if m: m.draw(output, str(i), 'c')