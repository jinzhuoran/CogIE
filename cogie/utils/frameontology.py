#
# class FrameOntology(Registrable):
#     """
#     This class is designed to read the ontology for FrameNet 1.x.
#     """
#
#     def __init__(self, data_path) -> None:
#         self._namespace = {"fn": "http://framenet.icsi.berkeley.edu"}
#         self._frame_index_filename = "frameIndex.xml"
#         self._frame_dir = "frame"
#
#         self.frames: Set[str] = set([])
#         self.frame_elements: Set[str] = set([])
#         self.lexical_units: Set[str] = set([])
#         self.frame_fe_map = dict()
#         self.core_frame_map = dict()
#         self.lu_frame_map = dict()
#         self.simple_lu_frame_map = dict()
#
#         self._read(data_path)
#         print("# frames in ontology: %d", len(self.frames))
#         print("# lexical units in ontology: %d", len(self.lexical_units))
#         print("# frame-elements in ontology: %d",
#               len(self.frame_elements))
#
#     def _simplify_lexunit(self, lexunit):
#
#         # situation: president_(political): president
#         if "_(" in lexunit:
#             speicial_flag_index = lexunit.index("_(")
#             simple_lu = lexunit[:speicial_flag_index]
#             return simple_lu
#
#         # situation: snow_event -> snow event
#         if not lexunit.isalpha():
#             speicial_flag_index = None
#             for i in range(len(lexunit)):
#                 if lexunit[i] != " " and lexunit[i].isalpha() != True:
#                     speicial_flag_index = i
#                     break
#             if speicial_flag_index:
#                 speicial_flag = lexunit[speicial_flag_index]
#                 split_lu_tokens = lexunit.split(speicial_flag)
#                 return " ".join(split_lu_tokens)
#
#         return lexunit
#
#     def _read_ontology_for_frame(self, frame_filename):
#         with open(frame_filename, "r", encoding="utf-8") as frame_file:
#             tree = ElementTree.parse(frame_file)
#         root = tree.getroot()
#
#         fe_for_frame = []
#         core_fe_list = []
#         for fe_tag in root.findall("fn:FE", self._namespace):
#             fe = fe_tag.attrib["name"]
#             fe_for_frame.append(fe)
#             self.frame_elements.add(fe)
#             if fe_tag.attrib["coreType"] == "Core":
#                 core_fe_list.append(fe)
#
#         lu_for_frame = [lu.attrib["name"].split(".")[0] for lu in root.findall("fn:lexUnit", self._namespace)]
#         for lu in lu_for_frame:
#             self.lexical_units.add(lu)
#
#         frame_file.close()
#         return fe_for_frame, core_fe_list, lu_for_frame
#
#     def _read_ontology(self, frame_index_filename: str) -> Set[str]:
#         print(frame_index_filename)
#         with open(frame_index_filename, "r", encoding="utf-8") as frame_file:
#             tree = ElementTree.parse(frame_file)
#         root = tree.getroot()
#
#         self.frames = set([frame.attrib["name"]
#                            for frame in root.findall("fn:frame", self._namespace)])
#
#     def _read(self, file_path: str):
#         frame_index_path = os.path.join(file_path, self._frame_index_filename)
#         print("Reading the frame ontology from %s", frame_index_path)
#         self._read_ontology(frame_index_path)
#
#         max_fe_for_frame = 0
#         total_fe_for_frame = 0.
#         max_core_fe_for_frame = 0
#         total_core_fe_for_frame = 0.
#         max_frames_for_lu = 0
#         total_frames_per_lu = 0.
#         longest_frame = None
#
#         frame_path = os.path.join(file_path, self._frame_dir)
#         print("Reading the frame-element - frame ontology from %s",
#               frame_path)
#         for frame in self.frames:
#             frame_file = os.path.join(frame_path, "{}.xml".format(frame))
#
#             fe_list, core_fe_list, lu_list = self._read_ontology_for_frame(
#                 frame_file)
#             self.frame_fe_map[frame] = fe_list
#             self.core_frame_map[frame] = core_fe_list
#
#             # Compute FE stats
#             total_fe_for_frame += len(self.frame_fe_map[frame])
#             if len(self.frame_fe_map[frame]) > max_fe_for_frame:
#                 max_fe_for_frame = len(self.frame_fe_map[frame])
#                 longest_frame = frame
#
#             # Compute core FE stats
#             total_core_fe_for_frame += len(self.core_frame_map[frame])
#             if len(self.core_frame_map[frame]) > max_core_fe_for_frame:
#                 max_core_fe_for_frame = len(self.core_frame_map[frame])
#
#             for lex_unit in lu_list:
#                 if lex_unit not in self.lu_frame_map:
#                     self.lu_frame_map[lex_unit] = []
#                 self.lu_frame_map[lex_unit].append(frame)
#
#                 simple_lex_unit = self._simplify_lexunit(lex_unit)
#                 if simple_lex_unit not in self.simple_lu_frame_map:
#                     self.simple_lu_frame_map[simple_lex_unit] = []
#                 self.simple_lu_frame_map[simple_lex_unit].append(frame)
#
#                 # Compute frame stats
#                 if len(self.lu_frame_map[lex_unit]) > max_frames_for_lu:
#                     max_frames_for_lu = len(self.lu_frame_map[lex_unit])
#                 total_frames_per_lu += len(self.lu_frame_map[lex_unit])
#
#         print("# max FEs per frame = %d (in frame %s)",
#               max_fe_for_frame, longest_frame)
#         print("# avg FEs per frame = %f",
#               total_fe_for_frame / len(self.frames))
#         print("# max core FEs per frame = %d", max_core_fe_for_frame)
#         print("# avg core FEs per frame = %f",
#               total_core_fe_for_frame / len(self.frames))
#         print("# max frames per LU = %d", max_frames_for_lu)
#         print("# avg frames per LU = %f",
#               total_frames_per_lu / len(self.lu_frame_map))