# -*-coding:utf-8-*-
import json
import copy
import numpy as np
from skimage.measure import points_in_poly


np.random.seed(0)


class Polygon(object):
    """
    Polygon represented as [N, 2] array of vertices
    """
    def __init__(self, name, vertices):
        """
        Initialize the polygon.

        Arguments:
            name: string, id of the polygon
            vertices: [N, 2] 2D numpy array of int
        """
        self._name = name
        self._vertices = vertices

    def __str__(self):
        return self._name

    def inside(self, coord):
        """
        Determine if a given coordinate is inside the polygon or not.

        Arguments:
            coord: 2 element tuple of int, e.g. (x, y)

        Returns:
            bool, if the coord is inside the polygon.
        """
        # print("type:",[coord],self._vertices.shape)
        return points_in_poly([coord], self._vertices)[0]

    def vertices(self):

        return np.array(self._vertices)

    def length(self):
        return self._vertices.shape[0]


class Annotation(object):
    """
    Annotation about the regions within WSI in terms of vertices of polygons.
    """
    def __init__(self):
        self._json_path = ''
        self._label_num2str ={
                    1: "other",
                    2: "cancer",
                    3: "normal_liver",
                    4: "hemorrhage_necrosis",
                    5: "cancer_beside",
                    6: "tertiary_lymphatic"
        }
        self._label_str2num = dict((v, k) for k, v in self._label_num2str.items())
        self._labelnum_polygons = {}
        for label_num in self._label_num2str:
            self._labelnum_polygons[label_num] = []

    def __str__(self):
        return self._json_path

    def from_json(self, json_path):
        """
        Initialize the annotation from a json file.

        Arguments:
            json_path: string, path to the json annotation.
        """
        self._json_path = json_path

        with open(json_path, 'r',encoding="utf-8") as f:
            annotations_json = json.load(f)
        cnt = -1
        Models =annotations_json["Models"]
        PolygonModel2 = Models["PolygonModel2"]
        for mark in PolygonModel2:
            cnt += 1
            if mark["Label"] in self._labelnum_polygons:
                coord = mark["Points"]
                vertices = []
                for value in coord:
                    if type(value) == int or len(value) == 2:
                        print("value:", value, self._json_path)
                        continue
                    vertices.append([int(round(value[0])), int(round(value[1]))])
                vertices = np.array(vertices)
                if len(vertices) == 0:
                    print("vertices is none:", vertices, self._json_path)
                if len(vertices) != 0:
                    polygon = Polygon(str(mark['Label']), vertices)
                    self._labelnum_polygons[mark["Label"]].append(polygon)
            else:
                # color->label
                print("Warning:unprocess classification!!! path and labelid:", self._json_path, mark['Label'])

    def inside_polygons(self, kind, coord):
        """
        Determine if a given coordinate is inside the tumor/tumor_bedide/
        fibrous_tissue/necrosis polygons of the annotation.

        Arguments:
            coord: 2 element tuple of int, e.g. (x, y)

        Returns:
            bool, if the coord is inside the positive/negative polygons of the
            annotation.
        """
        if kind in self._label_str2num:
            polygons = copy.deepcopy(self._labelnum_polygons[self._label_str2num[kind]])
        polygon_cp = []
        for polygon in polygons:
            if polygon.inside(coord):
                polygon_cp = polygon
                return True, polygon_cp
        return False, polygon_cp

    def polygon_vertices(self):
        """
        Return the polygon represented as [N, 2] array of vertices

        Arguments:
            is_positive: bool, return positive or negative polygons.

        Returns:
            [N, 2] 2D array of int
        """
        return list(map(lambda x: x.vertices(), self._polygons))

# class Formatter(object):
#     """
#     Format converter e.g. CAMELYON16 to internal json
#     """
#     def camelyon16xml2json(inxml, outjson):
#         """
#         Convert an annotation of camelyon16 xml format into a json format.
#
#         Arguments:
#             inxml: string, path to the input camelyon16 xml format
#             outjson: string, path to the output json format
#         """
#         root = ET.parse(inxml).getroot()
#         annotations_tumor = \
#             root.findall('./Annotations/Annotation[@PartOfGroup="Tumor"]')
#         annotations_0 = \
#             root.findall('./Annotations/Annotation[@PartOfGroup="_0"]')
#         annotations_1 = \
#             root.findall('./Annotations/Annotation[@PartOfGroup="_1"]')
#         annotations_2 = \
#             root.findall('./Annotations/Annotation[@PartOfGroup="_2"]')
#         annotations_positive = \
#             annotations_tumor + annotations_0 + annotations_1
#         annotations_negative = annotations_2
#
#         json_dict = {}
#         json_dict['positive'] = []
#         json_dict['negative'] = []
#
#         for annotation in annotations_positive:
#             X = list(map(lambda x: float(x.get('X')),
#                      annotation.findall('./Coordinates/Coordinate')))
#             Y = list(map(lambda x: float(x.get('Y')),
#                      annotation.findall('./Coordinates/Coordinate')))
#             vertices = np.round([X, Y]).astype(int).transpose().tolist()
#             name = annotation.attrib['Name']
#             json_dict['positive'].append({'name': name, 'vertices': vertices})
#
#         for annotation in annotations_negative:
#             X = list(map(lambda x: float(x.get('X')),
#                      annotation.findall('./Coordinates/Coordinate')))
#             Y = list(map(lambda x: float(x.get('Y')),
#                      annotation.findall('./Coordinates/Coordinate')))
#             vertices = np.round([X, Y]).astype(int).transpose().tolist()
#             name = annotation.attrib['Name']
#             json_dict['negative'].append({'name': name, 'vertices': vertices})
#
#         with open(outjson, 'w') as f:
#             json.dump(json_dict, f, indent=1)
#
#     def vertices2json(outjson, positive_vertices=[], negative_vertices=[]):
#         json_dict = {}
#         json_dict['positive'] = []
#         json_dict['negative'] = []
#
#         for i in range(len(positive_vertices)):
#             name = 'Annotation {}'.format(i)
#             vertices = positive_vertices[i].astype(int).tolist()
#             json_dict['positive'].append({'name': name, 'vertices': vertices})
#
#         for i in range(len(negative_vertices)):
#             name = 'Annotation {}'.format(i)
#             vertices = negative_vertices[i].astype(int).tolist()
#             json_dict['negative'].append({'name': name, 'vertices': vertices})
#
#         with open(outjson, 'w') as f:
#             json.dump(json_dict, f, indent=1)
