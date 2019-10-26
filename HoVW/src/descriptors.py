import cv2
import mahotas
import numpy as np

class ZernikeMoments:
    """The Zernike Moments descriptor invariant to scale, rotation and
    translation (interface for Mahotas Zernike moment's application).

    Parameters
    ----------
    degree: int, default = 8
        The degree of the polynomial moment (default corresponds 21
        polynoms).
    """

    def __init__(self, degree=8):
        self.degree = degree

    def describe(self, shape, radius):
        """Describe the Zernike Moments of self.degree order.

        Parameters
        ----------
        shape: Contour of shape to be described.
        radius: Radius of circumference which circumscribes the
            shape.

        Returns
        -------
        List of float
        The fist moments of order self.degree.
        """

        return mahotas.features.zernike_moments(shape, radius, self.degree)

class GeometricDescriptors:
    """Geometric descriptors. Implements multiple geometric descriptors
    statically.
    """
    @staticmethod
    def compactness(area, perimeter):
        """Calculate the Compactness of a shape.
            
            compactness = (4 * PI * Area) / (Perimeter ** 2)
            
        Parameters
        ----------
        area: area of the shape.
        perimeter: perimeter of the shape.
      
        Returns
        -------
        Float
        Compactness of the shape.
        """

        return (4*np.pi*area)/(perimeter**2)

    @staticmethod
    def convexity(contour, perimeter):
        """Calculate the Convexity of a shape.
            
            convexity = Perimeter of Hull / Perimeter
        
        Parameters
        ----------
        contour: contour of the shape.
        perimeter: perimeter of the shape.

        Returns
        -------
        Float
        Convexity of the shape.
        """

        phull = cv2.arcLength(cv2.convexHull(contour),True) #perimeter of hull

        return phull/perimeter

    @staticmethod
    def eccentricity(contour, centroid):
        """Calculate the Eccentricity of a shape.
        
        Parameters
        ----------
        contour: contour of the shape.
        centroid: perimeter of the shape.
        
        Returns
        -------
        Float
        Eccentricity of the shape.
        """

        def _covariance_matrix(contour, centroid):
            """Calculate the Covariance Matrix of a shape.
                 covariance marix = 1/N * 
                    sum(i=0;N-1) ([xi - cx][yi - cy]) *
                                ([xi - cx][yi - cy])^T =
                    ([cxx, cxy][cyx, cyy])
            
            cxy = cyx (happens beacuse of the derivation)
            
            Parameters
            ----------
            contour: contour of the shape.
            centroid: perimeter of the shape.
            
            Returns
            -------
            List of floats representing the covariance matrix of
                the shape.
            """

            cxx = cxy = cyy = 0.0
            for point in contour:
                point = point[0]
                l1 = point[0] - centroid[0]
                l2 = point[1] - centroid[1]
                cxx += l1*l1
                cxy += l1*l2
                cyy += l2*l2

            return [cxx, cxy, cyy]

        covM = _covariance_matrix(contour, centroid)
        bSqrt = np.sqrt((covM[0] + covM[2])*(covM[0] + covM[2])
                - 4 * (covM[0] * covM[2] - covM[1]*covM[1]))

        lambda1 = 1/2 * (covM[0] + covM[2] + bSqrt)
        lambda2 = 1/2 * (covM[0] + covM[2] - bSqrt)

        return lambda2/lambda1

    @staticmethod
    def ellipse_variance():
        raise NotImplementedError('ellipse_variance not implemented')

    @staticmethod
    def circle_variance():
        raise NotImplementedError('circle_variance not implemented')

    @staticmethod
    def rectangularity(area, contour):
        """Calculate the Rectangularity of a shape.
            
            rectangularity = area / minimum rectangle
        
        Parameters
        ----------
        area: area of the shape.
        contour: contour of the shape.
        
        Returns
        -------
        Float
        Rectangularity of the shape.
        """

        rectangle = cv2.minAreaRect(contour)

        return area/rectangle

    @staticmethod
    def avg_bending_energy(contour, step=1):
        """Calculate the Average Bending Energy of a shape.
            ref: https://stackoverflow.com/posts/34678359/revisions
        
        Parameters
        ----------
        contour: contour of the shape.
        step: ? (default: 1).
        
        Returns
        -------
        Float
        Average bending energy of the shape.
        """
        curvature = []
        if(len(contour) < step):
            return -1

        isClosed = max(np.absolute(contour[0][0]-contour[len(contour)-1][0])) <= 1
        for i in range(len(contour)):
            pos = contour[i][0]
            maxStep = step
            if(not isClosed):
                maxStep = min(min(step,i), len(contour)-1-i)
                if(maxStep == 0):
                    curvature2D = np.inf
                    curvature2D = 0
                    curvature.append(curvature2D)
                    continue

            iminus = i - maxStep
            iplus = i + maxStep
            pminus = contour[iminus + len(contour) if iminus < 0 else iminus][0]
            pplus = contour[iplus - len(contour) if iplus >= len(contour) else iplus][0]

            f1stDerivative = (pplus - pminus)/(iplus-iminus)
            f2ndDerivative = (pplus - 2*pos + pminus) / ((iplus-iminus)/2*(iplus-iminus)/2)

            divisor = f1stDerivative*f1stDerivative
            divisor = divisor[0] + divisor[1]
            if(np.absolute(divisor) > 10e-8):
                curvature2D = np.absolute(f1stDerivative[0]*f2ndDerivative[1] - f1stDerivative[1]*f2ndDerivative[0])
                curvature2D = curvature2D/pow(divisor, 3/2)
            else:
                curvature2D = np.inf
                curvature2D = 0

            curvature.append(curvature2D)

        return np.sum(curvature)/len(curvature)
