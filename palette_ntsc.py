import numpy as np

SRGB = {
  (0, 0): np.array((0, 0, 0)),
  (1, 0): np.array((0, 37, 0)),
  (2, 0): np.array((14, 14, 0)),
  (3, 0): np.array((0, 52, 0)),
  (4, 0): np.array((64, 0, 26)),
  (5, 0): np.array((0, 0, 0)),
  (6, 0): np.array((79, 0, 0)),
  (7, 0): np.array((14, 14, 0)),
  (8, 0): np.array((0, 0, 115)),
  (9, 0): np.array((0, 23, 88)),
  (10, 0): np.array((0, 0, 0)),
  (11, 0): np.array((0, 37, 0)),
  (12, 0): np.array((49, 0, 141)),
  (13, 0): np.array((0, 0, 115)),
  (14, 0): np.array((64, 0, 26)),
  (15, 0): np.array((0, 0, 0)),
  (16, 0): np.array((0, 101, 37)),
  (17, 0): np.array((0, 139, 11)),
  (18, 0): np.array((13, 116, 0)),
  (19, 0): np.array((0, 154, 0)),
  (20, 0): np.array((63, 63, 63)),
  (21, 0): np.array((0, 101, 37)),
  (22, 0): np.array((78, 78, 0)),
  (23, 0): np.array((13, 116, 0)),
  (24, 0): np.array((0, 86, 152)),
  (25, 0): np.array((0, 124, 126)),
  (26, 0): np.array((0, 101, 37)),
  (27, 0): np.array((0, 139, 11)),
  (28, 0): np.array((49, 48, 178)),
  (29, 0): np.array((0, 86, 152)),
  (30, 0): np.array((63, 63, 63)),
  (31, 0): np.array((0, 101, 37)),
  (32, 0): np.array((78, 78, 0)),
  (33, 0): np.array((13, 116, 0)),
  (34, 0): np.array((93, 93, 0)),
  (35, 0): np.array((28, 131, 0)),
  (36, 0): np.array((142, 40, 0)),
  (37, 0): np.array((78, 78, 0)),
  (38, 0): np.array((157, 55, 0)),
  (39, 0): np.array((93, 93, 0)),
  (40, 0): np.array((63, 63, 63)),
  (41, 0): np.array((0, 101, 37)),
  (42, 0): np.array((78, 78, 0)),
  (43, 0): np.array((13, 116, 0)),
  (44, 0): np.array((128, 25, 89)),
  (45, 0): np.array((63, 63, 63)),
  (46, 0): np.array((142, 40, 0)),
  (47, 0): np.array((78, 78, 0)),
  (48, 0): np.array((77, 180, 0)),
  (49, 0): np.array((13, 218, 0)),
  (50, 0): np.array((92, 195, 0)),
  (51, 0): np.array((27, 233, 0)),
  (52, 0): np.array((142, 142, 12)),
  (53, 0): np.array((77, 180, 0)),
  (54, 0): np.array((156, 157, 0)),
  (55, 0): np.array((92, 195, 0)),
  (56, 0): np.array((62, 165, 101)),
  (57, 0): np.array((0, 203, 75)),
  (58, 0): np.array((77, 180, 0)),
  (59, 0): np.array((13, 218, 0)),
  (60, 0): np.array((127, 127, 127)),
  (61, 0): np.array((62, 165, 101)),
  (62, 0): np.array((142, 142, 12)),
  (63, 0): np.array((77, 180, 0)),
  (64, 0): np.array((128, 25, 89)),
  (65, 0): np.array((63, 63, 63)),
  (66, 0): np.array((142, 40, 0)),
  (67, 0): np.array((78, 78, 0)),
  (68, 0): np.array((192, 0, 116)),
  (69, 0): np.array((128, 25, 89)),
  (70, 0): np.array((207, 2, 1)),
  (71, 0): np.array((142, 40, 0)),
  (72, 0): np.array((113, 10, 204)),
  (73, 0): np.array((49, 48, 178)),
  (74, 0): np.array((128, 25, 89)),
  (75, 0): np.array((63, 63, 63)),
  (76, 0): np.array((178, 0, 231)),
  (77, 0): np.array((113, 10, 204)),
  (78, 0): np.array((192, 0, 116)),
  (79, 0): np.array((128, 25, 89)),
  (80, 0): np.array((127, 127, 127)),
  (81, 0): np.array((62, 165, 101)),
  (82, 0): np.array((142, 142, 12)),
  (83, 0): np.array((77, 180, 0)),
  (84, 0): np.array((192, 89, 153)),
  (85, 0): np.array((127, 127, 127)),
  (86, 0): np.array((206, 104, 38)),
  (87, 0): np.array((142, 142, 12)),
  (88, 0): np.array((112, 112, 242)),
  (89, 0): np.array((48, 150, 216)),
  (90, 0): np.array((127, 127, 127)),
  (91, 0): np.array((62, 165, 101)),
  (92, 0): np.array((177, 74, 255)),
  (93, 0): np.array((112, 112, 242)),
  (94, 0): np.array((192, 89, 153)),
  (95, 0): np.array((127, 127, 127)),
  (96, 0): np.array((206, 104, 38)),
  (97, 0): np.array((142, 142, 12)),
  (98, 0): np.array((221, 119, 0)),
  (99, 0): np.array((156, 157, 0)),
  (100, 0): np.array((255, 66, 64)),
  (101, 0): np.array((206, 104, 38)),
  (102, 0): np.array((255, 81, 0)),
  (103, 0): np.array((221, 119, 0)),
  (104, 0): np.array((192, 89, 153)),
  (105, 0): np.array((127, 127, 127)),
  (106, 0): np.array((206, 104, 38)),
  (107, 0): np.array((142, 142, 12)),
  (108, 0): np.array((255, 51, 179)),
  (109, 0): np.array((192, 89, 153)),
  (110, 0): np.array((255, 66, 64)),
  (111, 0): np.array((206, 104, 38)),
  (112, 0): np.array((205, 206, 76)),
  (113, 0): np.array((141, 244, 50)),
  (114, 0): np.array((220, 220, 0)),
  (115, 0): np.array((156, 255, 0)),
  (116, 0): np.array((255, 168, 102)),
  (117, 0): np.array((205, 206, 76)),
  (118, 0): np.array((255, 183, 0)),
  (119, 0): np.array((220, 220, 0)),
  (120, 0): np.array((191, 191, 191)),
  (121, 0): np.array((126, 229, 165)),
  (122, 0): np.array((205, 206, 76)),
  (123, 0): np.array((141, 244, 50)),
  (124, 0): np.array((255, 153, 217)),
  (125, 0): np.array((191, 191, 191)),
  (126, 0): np.array((255, 168, 102)),
  (127, 0): np.array((205, 206, 76)),
  (128, 0): np.array((49, 48, 178)),
  (129, 0): np.array((0, 86, 152)),
  (130, 0): np.array((63, 63, 63)),
  (131, 0): np.array((0, 101, 37)),
  (132, 0): np.array((113, 10, 204)),
  (133, 0): np.array((49, 48, 178)),
  (134, 0): np.array((128, 25, 89)),
  (135, 0): np.array((63, 63, 63)),
  (136, 0): np.array((34, 34, 255)),
  (137, 0): np.array((0, 71, 255)),
  (138, 0): np.array((49, 48, 178)),
  (139, 0): np.array((0, 86, 152)),
  (140, 0): np.array((98, 0, 255)),
  (141, 0): np.array((34, 34, 255)),
  (142, 0): np.array((113, 10, 204)),
  (143, 0): np.array((49, 48, 178)),
  (144, 0): np.array((48, 150, 216)),
  (145, 0): np.array((0, 188, 190)),
  (146, 0): np.array((62, 165, 101)),
  (147, 0): np.array((0, 203, 75)),
  (148, 0): np.array((112, 112, 242)),
  (149, 0): np.array((48, 150, 216)),
  (150, 0): np.array((127, 127, 127)),
  (151, 0): np.array((62, 165, 101)),
  (152, 0): np.array((33, 135, 255)),
  (153, 0): np.array((0, 173, 255)),
  (154, 0): np.array((48, 150, 216)),
  (155, 0): np.array((0, 188, 190)),
  (156, 0): np.array((98, 97, 255)),
  (157, 0): np.array((33, 135, 255)),
  (158, 0): np.array((112, 112, 242)),
  (159, 0): np.array((48, 150, 216)),
  (160, 0): np.array((127, 127, 127)),
  (161, 0): np.array((62, 165, 101)),
  (162, 0): np.array((142, 142, 12)),
  (163, 0): np.array((77, 180, 0)),
  (164, 0): np.array((192, 89, 153)),
  (165, 0): np.array((127, 127, 127)),
  (166, 0): np.array((206, 104, 38)),
  (167, 0): np.array((142, 142, 12)),
  (168, 0): np.array((112, 112, 242)),
  (169, 0): np.array((48, 150, 216)),
  (170, 0): np.array((127, 127, 127)),
  (171, 0): np.array((62, 165, 101)),
  (172, 0): np.array((177, 74, 255)),
  (173, 0): np.array((112, 112, 242)),
  (174, 0): np.array((192, 89, 153)),
  (175, 0): np.array((127, 127, 127)),
  (176, 0): np.array((126, 229, 165)),
  (177, 0): np.array((62, 255, 138)),
  (178, 0): np.array((141, 244, 50)),
  (179, 0): np.array((76, 255, 23)),
  (180, 0): np.array((191, 191, 191)),
  (181, 0): np.array((126, 229, 165)),
  (182, 0): np.array((205, 206, 76)),
  (183, 0): np.array((141, 244, 50)),
  (184, 0): np.array((112, 214, 255)),
  (185, 0): np.array((47, 252, 253)),
  (186, 0): np.array((126, 229, 165)),
  (187, 0): np.array((62, 255, 138)),
  (188, 0): np.array((176, 176, 255)),
  (189, 0): np.array((112, 214, 255)),
  (190, 0): np.array((191, 191, 191)),
  (191, 0): np.array((126, 229, 165)),
  (192, 0): np.array((177, 74, 255)),
  (193, 0): np.array((112, 112, 242)),
  (194, 0): np.array((192, 89, 153)),
  (195, 0): np.array((127, 127, 127)),
  (196, 0): np.array((241, 36, 255)),
  (197, 0): np.array((177, 74, 255)),
  (198, 0): np.array((255, 51, 179)),
  (199, 0): np.array((192, 89, 153)),
  (200, 0): np.array((162, 59, 255)),
  (201, 0): np.array((98, 97, 255)),
  (202, 0): np.array((177, 74, 255)),
  (203, 0): np.array((112, 112, 242)),
  (204, 0): np.array((227, 21, 255)),
  (205, 0): np.array((162, 59, 255)),
  (206, 0): np.array((241, 36, 255)),
  (207, 0): np.array((177, 74, 255)),
  (208, 0): np.array((176, 176, 255)),
  (209, 0): np.array((112, 214, 255)),
  (210, 0): np.array((191, 191, 191)),
  (211, 0): np.array((126, 229, 165)),
  (212, 0): np.array((241, 138, 255)),
  (213, 0): np.array((176, 176, 255)),
  (214, 0): np.array((255, 153, 217)),
  (215, 0): np.array((191, 191, 191)),
  (216, 0): np.array((161, 161, 255)),
  (217, 0): np.array((97, 199, 255)),
  (218, 0): np.array((176, 176, 255)),
  (219, 0): np.array((112, 214, 255)),
  (220, 0): np.array((226, 123, 255)),
  (221, 0): np.array((161, 161, 255)),
  (222, 0): np.array((241, 138, 255)),
  (223, 0): np.array((176, 176, 255)),
  (224, 0): np.array((255, 153, 217)),
  (225, 0): np.array((191, 191, 191)),
  (226, 0): np.array((255, 168, 102)),
  (227, 0): np.array((205, 206, 76)),
  (228, 0): np.array((255, 115, 243)),
  (229, 0): np.array((255, 153, 217)),
  (230, 0): np.array((255, 130, 128)),
  (231, 0): np.array((255, 168, 102)),
  (232, 0): np.array((241, 138, 255)),
  (233, 0): np.array((176, 176, 255)),
  (234, 0): np.array((255, 153, 217)),
  (235, 0): np.array((191, 191, 191)),
  (236, 0): np.array((255, 100, 255)),
  (237, 0): np.array((241, 138, 255)),
  (238, 0): np.array((255, 115, 243)),
  (239, 0): np.array((255, 153, 217)),
  (240, 0): np.array((255, 255, 255)),
  (241, 0): np.array((190, 255, 228)),
  (242, 0): np.array((255, 255, 139)),
  (243, 0): np.array((205, 255, 113)),
  (244, 0): np.array((255, 217, 255)),
  (245, 0): np.array((254, 254, 255)),
  (246, 0): np.array((255, 231, 166)),
  (247, 0): np.array((255, 255, 139)),
  (248, 0): np.array((240, 240, 255)),
  (249, 0): np.array((175, 255, 255)),
  (250, 0): np.array((254, 255, 255)),
  (251, 0): np.array((190, 255, 228)),
  (252, 0): np.array((255, 202, 255)),
  (253, 0): np.array((240, 240, 255)),
  (254, 0): np.array((255, 217, 255)),
  (255, 0): np.array((254, 254, 255)),
  (0, 1): np.array((0, 0, 0)),
  (1, 1): np.array((14, 14, 0)),
  (2, 1): np.array((64, 0, 26)),
  (3, 1): np.array((79, 0, 0)),
  (4, 1): np.array((0, 0, 115)),
  (5, 1): np.array((0, 0, 0)),
  (6, 1): np.array((49, 0, 141)),
  (7, 1): np.array((64, 0, 26)),
  (8, 1): np.array((0, 37, 0)),
  (9, 1): np.array((0, 52, 0)),
  (10, 1): np.array((0, 0, 0)),
  (11, 1): np.array((14, 14, 0)),
  (12, 1): np.array((0, 23, 88)),
  (13, 1): np.array((0, 37, 0)),
  (14, 1): np.array((0, 0, 115)),
  (15, 1): np.array((0, 0, 0)),
  (16, 1): np.array((78, 78, 0)),
  (17, 1): np.array((93, 93, 0)),
  (18, 1): np.array((142, 40, 0)),
  (19, 1): np.array((157, 55, 0)),
  (20, 1): np.array((63, 63, 63)),
  (21, 1): np.array((78, 78, 0)),
  (22, 1): np.array((128, 25, 89)),
  (23, 1): np.array((142, 40, 0)),
  (24, 1): np.array((13, 116, 0)),
  (25, 1): np.array((28, 131, 0)),
  (26, 1): np.array((78, 78, 0)),
  (27, 1): np.array((93, 93, 0)),
  (28, 1): np.array((0, 101, 37)),
  (29, 1): np.array((13, 116, 0)),
  (30, 1): np.array((63, 63, 63)),
  (31, 1): np.array((78, 78, 0)),
  (32, 1): np.array((128, 25, 89)),
  (33, 1): np.array((142, 40, 0)),
  (34, 1): np.array((192, 0, 116)),
  (35, 1): np.array((207, 2, 1)),
  (36, 1): np.array((113, 10, 204)),
  (37, 1): np.array((128, 25, 89)),
  (38, 1): np.array((178, 0, 231)),
  (39, 1): np.array((192, 0, 116)),
  (40, 1): np.array((63, 63, 63)),
  (41, 1): np.array((78, 78, 0)),
  (42, 1): np.array((128, 25, 89)),
  (43, 1): np.array((142, 40, 0)),
  (44, 1): np.array((49, 48, 178)),
  (45, 1): np.array((63, 63, 63)),
  (46, 1): np.array((113, 10, 204)),
  (47, 1): np.array((128, 25, 89)),
  (48, 1): np.array((206, 104, 38)),
  (49, 1): np.array((221, 119, 0)),
  (50, 1): np.array((255, 66, 64)),
  (51, 1): np.array((255, 81, 0)),
  (52, 1): np.array((192, 89, 153)),
  (53, 1): np.array((206, 104, 38)),
  (54, 1): np.array((255, 51, 179)),
  (55, 1): np.array((255, 66, 64)),
  (56, 1): np.array((142, 142, 12)),
  (57, 1): np.array((156, 157, 0)),
  (58, 1): np.array((206, 104, 38)),
  (59, 1): np.array((221, 119, 0)),
  (60, 1): np.array((127, 127, 127)),
  (61, 1): np.array((142, 142, 12)),
  (62, 1): np.array((192, 89, 153)),
  (63, 1): np.array((206, 104, 38)),
  (64, 1): np.array((49, 48, 178)),
  (65, 1): np.array((63, 63, 63)),
  (66, 1): np.array((113, 10, 204)),
  (67, 1): np.array((128, 25, 89)),
  (68, 1): np.array((34, 34, 255)),
  (69, 1): np.array((49, 48, 178)),
  (70, 1): np.array((98, 0, 255)),
  (71, 1): np.array((113, 10, 204)),
  (72, 1): np.array((0, 86, 152)),
  (73, 1): np.array((0, 101, 37)),
  (74, 1): np.array((49, 48, 178)),
  (75, 1): np.array((63, 63, 63)),
  (76, 1): np.array((0, 71, 255)),
  (77, 1): np.array((0, 86, 152)),
  (78, 1): np.array((34, 34, 255)),
  (79, 1): np.array((49, 48, 178)),
  (80, 1): np.array((127, 127, 127)),
  (81, 1): np.array((142, 142, 12)),
  (82, 1): np.array((192, 89, 153)),
  (83, 1): np.array((206, 104, 38)),
  (84, 1): np.array((112, 112, 242)),
  (85, 1): np.array((127, 127, 127)),
  (86, 1): np.array((177, 74, 255)),
  (87, 1): np.array((192, 89, 153)),
  (88, 1): np.array((62, 165, 101)),
  (89, 1): np.array((77, 180, 0)),
  (90, 1): np.array((127, 127, 127)),
  (91, 1): np.array((142, 142, 12)),
  (92, 1): np.array((48, 150, 216)),
  (93, 1): np.array((62, 165, 101)),
  (94, 1): np.array((112, 112, 242)),
  (95, 1): np.array((127, 127, 127)),
  (96, 1): np.array((177, 74, 255)),
  (97, 1): np.array((192, 89, 153)),
  (98, 1): np.array((241, 36, 255)),
  (99, 1): np.array((255, 51, 179)),
  (100, 1): np.array((162, 59, 255)),
  (101, 1): np.array((177, 74, 255)),
  (102, 1): np.array((227, 21, 255)),
  (103, 1): np.array((241, 36, 255)),
  (104, 1): np.array((112, 112, 242)),
  (105, 1): np.array((127, 127, 127)),
  (106, 1): np.array((177, 74, 255)),
  (107, 1): np.array((192, 89, 153)),
  (108, 1): np.array((98, 97, 255)),
  (109, 1): np.array((112, 112, 242)),
  (110, 1): np.array((162, 59, 255)),
  (111, 1): np.array((177, 74, 255)),
  (112, 1): np.array((255, 153, 217)),
  (113, 1): np.array((255, 168, 102)),
  (114, 1): np.array((255, 115, 243)),
  (115, 1): np.array((255, 130, 128)),
  (116, 1): np.array((241, 138, 255)),
  (117, 1): np.array((255, 153, 217)),
  (118, 1): np.array((255, 100, 255)),
  (119, 1): np.array((255, 115, 243)),
  (120, 1): np.array((191, 191, 191)),
  (121, 1): np.array((205, 206, 76)),
  (122, 1): np.array((255, 153, 217)),
  (123, 1): np.array((255, 168, 102)),
  (124, 1): np.array((176, 176, 255)),
  (125, 1): np.array((191, 191, 191)),
  (126, 1): np.array((241, 138, 255)),
  (127, 1): np.array((255, 153, 217)),
  (128, 1): np.array((0, 101, 37)),
  (129, 1): np.array((13, 116, 0)),
  (130, 1): np.array((63, 63, 63)),
  (131, 1): np.array((78, 78, 0)),
  (132, 1): np.array((0, 86, 152)),
  (133, 1): np.array((0, 101, 37)),
  (134, 1): np.array((49, 48, 178)),
  (135, 1): np.array((63, 63, 63)),
  (136, 1): np.array((0, 139, 11)),
  (137, 1): np.array((0, 154, 0)),
  (138, 1): np.array((0, 101, 37)),
  (139, 1): np.array((13, 116, 0)),
  (140, 1): np.array((0, 124, 126)),
  (141, 1): np.array((0, 139, 11)),
  (142, 1): np.array((0, 86, 152)),
  (143, 1): np.array((0, 101, 37)),
  (144, 1): np.array((77, 180, 0)),
  (145, 1): np.array((92, 195, 0)),
  (146, 1): np.array((142, 142, 12)),
  (147, 1): np.array((156, 157, 0)),
  (148, 1): np.array((62, 165, 101)),
  (149, 1): np.array((77, 180, 0)),
  (150, 1): np.array((127, 127, 127)),
  (151, 1): np.array((142, 142, 12)),
  (152, 1): np.array((13, 218, 0)),
  (153, 1): np.array((27, 233, 0)),
  (154, 1): np.array((77, 180, 0)),
  (155, 1): np.array((92, 195, 0)),
  (156, 1): np.array((0, 203, 75)),
  (157, 1): np.array((13, 218, 0)),
  (158, 1): np.array((62, 165, 101)),
  (159, 1): np.array((77, 180, 0)),
  (160, 1): np.array((127, 127, 127)),
  (161, 1): np.array((142, 142, 12)),
  (162, 1): np.array((192, 89, 153)),
  (163, 1): np.array((206, 104, 38)),
  (164, 1): np.array((112, 112, 242)),
  (165, 1): np.array((127, 127, 127)),
  (166, 1): np.array((177, 74, 255)),
  (167, 1): np.array((192, 89, 153)),
  (168, 1): np.array((62, 165, 101)),
  (169, 1): np.array((77, 180, 0)),
  (170, 1): np.array((127, 127, 127)),
  (171, 1): np.array((142, 142, 12)),
  (172, 1): np.array((48, 150, 216)),
  (173, 1): np.array((62, 165, 101)),
  (174, 1): np.array((112, 112, 242)),
  (175, 1): np.array((127, 127, 127)),
  (176, 1): np.array((205, 206, 76)),
  (177, 1): np.array((220, 220, 0)),
  (178, 1): np.array((255, 168, 102)),
  (179, 1): np.array((255, 183, 0)),
  (180, 1): np.array((191, 191, 191)),
  (181, 1): np.array((205, 206, 76)),
  (182, 1): np.array((255, 153, 217)),
  (183, 1): np.array((255, 168, 102)),
  (184, 1): np.array((141, 244, 50)),
  (185, 1): np.array((156, 255, 0)),
  (186, 1): np.array((205, 206, 76)),
  (187, 1): np.array((220, 220, 0)),
  (188, 1): np.array((126, 229, 165)),
  (189, 1): np.array((141, 244, 50)),
  (190, 1): np.array((191, 191, 191)),
  (191, 1): np.array((205, 206, 76)),
  (192, 1): np.array((48, 150, 216)),
  (193, 1): np.array((62, 165, 101)),
  (194, 1): np.array((112, 112, 242)),
  (195, 1): np.array((127, 127, 127)),
  (196, 1): np.array((33, 135, 255)),
  (197, 1): np.array((48, 150, 216)),
  (198, 1): np.array((98, 97, 255)),
  (199, 1): np.array((112, 112, 242)),
  (200, 1): np.array((0, 188, 190)),
  (201, 1): np.array((0, 203, 75)),
  (202, 1): np.array((48, 150, 216)),
  (203, 1): np.array((62, 165, 101)),
  (204, 1): np.array((0, 173, 255)),
  (205, 1): np.array((0, 188, 190)),
  (206, 1): np.array((33, 135, 255)),
  (207, 1): np.array((48, 150, 216)),
  (208, 1): np.array((126, 229, 165)),
  (209, 1): np.array((141, 244, 50)),
  (210, 1): np.array((191, 191, 191)),
  (211, 1): np.array((205, 206, 76)),
  (212, 1): np.array((112, 214, 255)),
  (213, 1): np.array((126, 229, 165)),
  (214, 1): np.array((176, 176, 255)),
  (215, 1): np.array((191, 191, 191)),
  (216, 1): np.array((62, 255, 138)),
  (217, 1): np.array((76, 255, 23)),
  (218, 1): np.array((126, 229, 165)),
  (219, 1): np.array((141, 244, 50)),
  (220, 1): np.array((47, 252, 253)),
  (221, 1): np.array((62, 255, 138)),
  (222, 1): np.array((112, 214, 255)),
  (223, 1): np.array((126, 229, 165)),
  (224, 1): np.array((176, 176, 255)),
  (225, 1): np.array((191, 191, 191)),
  (226, 1): np.array((241, 138, 255)),
  (227, 1): np.array((255, 153, 217)),
  (228, 1): np.array((161, 161, 255)),
  (229, 1): np.array((176, 176, 255)),
  (230, 1): np.array((226, 123, 255)),
  (231, 1): np.array((241, 138, 255)),
  (232, 1): np.array((112, 214, 255)),
  (233, 1): np.array((126, 229, 165)),
  (234, 1): np.array((176, 176, 255)),
  (235, 1): np.array((191, 191, 191)),
  (236, 1): np.array((97, 199, 255)),
  (237, 1): np.array((112, 214, 255)),
  (238, 1): np.array((161, 161, 255)),
  (239, 1): np.array((176, 176, 255)),
  (240, 1): np.array((255, 255, 255)),
  (241, 1): np.array((255, 255, 139)),
  (242, 1): np.array((255, 217, 255)),
  (243, 1): np.array((255, 231, 166)),
  (244, 1): np.array((240, 240, 255)),
  (245, 1): np.array((254, 255, 255)),
  (246, 1): np.array((255, 202, 255)),
  (247, 1): np.array((255, 217, 255)),
  (248, 1): np.array((190, 255, 228)),
  (249, 1): np.array((205, 255, 113)),
  (250, 1): np.array((254, 255, 255)),
  (251, 1): np.array((255, 255, 139)),
  (252, 1): np.array((175, 255, 255)),
  (253, 1): np.array((190, 255, 228)),
  (254, 1): np.array((240, 240, 255)),
  (255, 1): np.array((254, 255, 255)),
  (0, 2): np.array((0, 0, 0)),
  (1, 2): np.array((64, 0, 26)),
  (2, 2): np.array((0, 0, 115)),
  (3, 2): np.array((49, 0, 141)),
  (4, 2): np.array((0, 37, 0)),
  (5, 2): np.array((0, 0, 0)),
  (6, 2): np.array((0, 23, 88)),
  (7, 2): np.array((0, 0, 115)),
  (8, 2): np.array((14, 14, 0)),
  (9, 2): np.array((79, 0, 0)),
  (10, 2): np.array((0, 0, 0)),
  (11, 2): np.array((64, 0, 26)),
  (12, 2): np.array((0, 52, 0)),
  (13, 2): np.array((14, 14, 0)),
  (14, 2): np.array((0, 37, 0)),
  (15, 2): np.array((0, 0, 0)),
  (16, 2): np.array((128, 25, 89)),
  (17, 2): np.array((192, 0, 116)),
  (18, 2): np.array((113, 10, 204)),
  (19, 2): np.array((178, 0, 231)),
  (20, 2): np.array((63, 63, 63)),
  (21, 2): np.array((128, 25, 89)),
  (22, 2): np.array((49, 48, 178)),
  (23, 2): np.array((113, 10, 204)),
  (24, 2): np.array((142, 40, 0)),
  (25, 2): np.array((207, 2, 1)),
  (26, 2): np.array((128, 25, 89)),
  (27, 2): np.array((192, 0, 116)),
  (28, 2): np.array((78, 78, 0)),
  (29, 2): np.array((142, 40, 0)),
  (30, 2): np.array((63, 63, 63)),
  (31, 2): np.array((128, 25, 89)),
  (32, 2): np.array((49, 48, 178)),
  (33, 2): np.array((113, 10, 204)),
  (34, 2): np.array((34, 34, 255)),
  (35, 2): np.array((98, 0, 255)),
  (36, 2): np.array((0, 86, 152)),
  (37, 2): np.array((49, 48, 178)),
  (38, 2): np.array((0, 71, 255)),
  (39, 2): np.array((34, 34, 255)),
  (40, 2): np.array((63, 63, 63)),
  (41, 2): np.array((128, 25, 89)),
  (42, 2): np.array((49, 48, 178)),
  (43, 2): np.array((113, 10, 204)),
  (44, 2): np.array((0, 101, 37)),
  (45, 2): np.array((63, 63, 63)),
  (46, 2): np.array((0, 86, 152)),
  (47, 2): np.array((49, 48, 178)),
  (48, 2): np.array((177, 74, 255)),
  (49, 2): np.array((241, 36, 255)),
  (50, 2): np.array((162, 59, 255)),
  (51, 2): np.array((227, 21, 255)),
  (52, 2): np.array((112, 112, 242)),
  (53, 2): np.array((177, 74, 255)),
  (54, 2): np.array((98, 97, 255)),
  (55, 2): np.array((162, 59, 255)),
  (56, 2): np.array((192, 89, 153)),
  (57, 2): np.array((255, 51, 179)),
  (58, 2): np.array((177, 74, 255)),
  (59, 2): np.array((241, 36, 255)),
  (60, 2): np.array((127, 127, 127)),
  (61, 2): np.array((192, 89, 153)),
  (62, 2): np.array((112, 112, 242)),
  (63, 2): np.array((177, 74, 255)),
  (64, 2): np.array((0, 101, 37)),
  (65, 2): np.array((63, 63, 63)),
  (66, 2): np.array((0, 86, 152)),
  (67, 2): np.array((49, 48, 178)),
  (68, 2): np.array((0, 139, 11)),
  (69, 2): np.array((0, 101, 37)),
  (70, 2): np.array((0, 124, 126)),
  (71, 2): np.array((0, 86, 152)),
  (72, 2): np.array((13, 116, 0)),
  (73, 2): np.array((78, 78, 0)),
  (74, 2): np.array((0, 101, 37)),
  (75, 2): np.array((63, 63, 63)),
  (76, 2): np.array((0, 154, 0)),
  (77, 2): np.array((13, 116, 0)),
  (78, 2): np.array((0, 139, 11)),
  (79, 2): np.array((0, 101, 37)),
  (80, 2): np.array((127, 127, 127)),
  (81, 2): np.array((192, 89, 153)),
  (82, 2): np.array((112, 112, 242)),
  (83, 2): np.array((177, 74, 255)),
  (84, 2): np.array((62, 165, 101)),
  (85, 2): np.array((127, 127, 127)),
  (86, 2): np.array((48, 150, 216)),
  (87, 2): np.array((112, 112, 242)),
  (88, 2): np.array((142, 142, 12)),
  (89, 2): np.array((206, 104, 38)),
  (90, 2): np.array((127, 127, 127)),
  (91, 2): np.array((192, 89, 153)),
  (92, 2): np.array((77, 180, 0)),
  (93, 2): np.array((142, 142, 12)),
  (94, 2): np.array((62, 165, 101)),
  (95, 2): np.array((127, 127, 127)),
  (96, 2): np.array((48, 150, 216)),
  (97, 2): np.array((112, 112, 242)),
  (98, 2): np.array((33, 135, 255)),
  (99, 2): np.array((98, 97, 255)),
  (100, 2): np.array((0, 188, 190)),
  (101, 2): np.array((48, 150, 216)),
  (102, 2): np.array((0, 173, 255)),
  (103, 2): np.array((33, 135, 255)),
  (104, 2): np.array((62, 165, 101)),
  (105, 2): np.array((127, 127, 127)),
  (106, 2): np.array((48, 150, 216)),
  (107, 2): np.array((112, 112, 242)),
  (108, 2): np.array((0, 203, 75)),
  (109, 2): np.array((62, 165, 101)),
  (110, 2): np.array((0, 188, 190)),
  (111, 2): np.array((48, 150, 216)),
  (112, 2): np.array((176, 176, 255)),
  (113, 2): np.array((241, 138, 255)),
  (114, 2): np.array((161, 161, 255)),
  (115, 2): np.array((226, 123, 255)),
  (116, 2): np.array((112, 214, 255)),
  (117, 2): np.array((176, 176, 255)),
  (118, 2): np.array((97, 199, 255)),
  (119, 2): np.array((161, 161, 255)),
  (120, 2): np.array((191, 191, 191)),
  (121, 2): np.array((255, 153, 217)),
  (122, 2): np.array((176, 176, 255)),
  (123, 2): np.array((241, 138, 255)),
  (124, 2): np.array((126, 229, 165)),
  (125, 2): np.array((191, 191, 191)),
  (126, 2): np.array((112, 214, 255)),
  (127, 2): np.array((176, 176, 255)),
  (128, 2): np.array((78, 78, 0)),
  (129, 2): np.array((142, 40, 0)),
  (130, 2): np.array((63, 63, 63)),
  (131, 2): np.array((128, 25, 89)),
  (132, 2): np.array((13, 116, 0)),
  (133, 2): np.array((78, 78, 0)),
  (134, 2): np.array((0, 101, 37)),
  (135, 2): np.array((63, 63, 63)),
  (136, 2): np.array((93, 93, 0)),
  (137, 2): np.array((157, 55, 0)),
  (138, 2): np.array((78, 78, 0)),
  (139, 2): np.array((142, 40, 0)),
  (140, 2): np.array((28, 131, 0)),
  (141, 2): np.array((93, 93, 0)),
  (142, 2): np.array((13, 116, 0)),
  (143, 2): np.array((78, 78, 0)),
  (144, 2): np.array((206, 104, 38)),
  (145, 2): np.array((255, 66, 64)),
  (146, 2): np.array((192, 89, 153)),
  (147, 2): np.array((255, 51, 179)),
  (148, 2): np.array((142, 142, 12)),
  (149, 2): np.array((206, 104, 38)),
  (150, 2): np.array((127, 127, 127)),
  (151, 2): np.array((192, 89, 153)),
  (152, 2): np.array((221, 119, 0)),
  (153, 2): np.array((255, 81, 0)),
  (154, 2): np.array((206, 104, 38)),
  (155, 2): np.array((255, 66, 64)),
  (156, 2): np.array((156, 157, 0)),
  (157, 2): np.array((221, 119, 0)),
  (158, 2): np.array((142, 142, 12)),
  (159, 2): np.array((206, 104, 38)),
  (160, 2): np.array((127, 127, 127)),
  (161, 2): np.array((192, 89, 153)),
  (162, 2): np.array((112, 112, 242)),
  (163, 2): np.array((177, 74, 255)),
  (164, 2): np.array((62, 165, 101)),
  (165, 2): np.array((127, 127, 127)),
  (166, 2): np.array((48, 150, 216)),
  (167, 2): np.array((112, 112, 242)),
  (168, 2): np.array((142, 142, 12)),
  (169, 2): np.array((206, 104, 38)),
  (170, 2): np.array((127, 127, 127)),
  (171, 2): np.array((192, 89, 153)),
  (172, 2): np.array((77, 180, 0)),
  (173, 2): np.array((142, 142, 12)),
  (174, 2): np.array((62, 165, 101)),
  (175, 2): np.array((127, 127, 127)),
  (176, 2): np.array((255, 153, 217)),
  (177, 2): np.array((255, 115, 243)),
  (178, 2): np.array((241, 138, 255)),
  (179, 2): np.array((255, 100, 255)),
  (180, 2): np.array((191, 191, 191)),
  (181, 2): np.array((255, 153, 217)),
  (182, 2): np.array((176, 176, 255)),
  (183, 2): np.array((241, 138, 255)),
  (184, 2): np.array((255, 168, 102)),
  (185, 2): np.array((255, 130, 128)),
  (186, 2): np.array((255, 153, 217)),
  (187, 2): np.array((255, 115, 243)),
  (188, 2): np.array((205, 206, 76)),
  (189, 2): np.array((255, 168, 102)),
  (190, 2): np.array((191, 191, 191)),
  (191, 2): np.array((255, 153, 217)),
  (192, 2): np.array((77, 180, 0)),
  (193, 2): np.array((142, 142, 12)),
  (194, 2): np.array((62, 165, 101)),
  (195, 2): np.array((127, 127, 127)),
  (196, 2): np.array((13, 218, 0)),
  (197, 2): np.array((77, 180, 0)),
  (198, 2): np.array((0, 203, 75)),
  (199, 2): np.array((62, 165, 101)),
  (200, 2): np.array((92, 195, 0)),
  (201, 2): np.array((156, 157, 0)),
  (202, 2): np.array((77, 180, 0)),
  (203, 2): np.array((142, 142, 12)),
  (204, 2): np.array((27, 233, 0)),
  (205, 2): np.array((92, 195, 0)),
  (206, 2): np.array((13, 218, 0)),
  (207, 2): np.array((77, 180, 0)),
  (208, 2): np.array((205, 206, 76)),
  (209, 2): np.array((255, 168, 102)),
  (210, 2): np.array((191, 191, 191)),
  (211, 2): np.array((255, 153, 217)),
  (212, 2): np.array((141, 244, 50)),
  (213, 2): np.array((205, 206, 76)),
  (214, 2): np.array((126, 229, 165)),
  (215, 2): np.array((191, 191, 191)),
  (216, 2): np.array((220, 220, 0)),
  (217, 2): np.array((255, 183, 0)),
  (218, 2): np.array((205, 206, 76)),
  (219, 2): np.array((255, 168, 102)),
  (220, 2): np.array((156, 255, 0)),
  (221, 2): np.array((220, 220, 0)),
  (222, 2): np.array((141, 244, 50)),
  (223, 2): np.array((205, 206, 76)),
  (224, 2): np.array((126, 229, 165)),
  (225, 2): np.array((191, 191, 191)),
  (226, 2): np.array((112, 214, 255)),
  (227, 2): np.array((176, 176, 255)),
  (228, 2): np.array((62, 255, 138)),
  (229, 2): np.array((126, 229, 165)),
  (230, 2): np.array((47, 252, 253)),
  (231, 2): np.array((112, 214, 255)),
  (232, 2): np.array((141, 244, 50)),
  (233, 2): np.array((205, 206, 76)),
  (234, 2): np.array((126, 229, 165)),
  (235, 2): np.array((191, 191, 191)),
  (236, 2): np.array((76, 255, 23)),
  (237, 2): np.array((141, 244, 50)),
  (238, 2): np.array((62, 255, 138)),
  (239, 2): np.array((126, 229, 165)),
  (240, 2): np.array((254, 255, 255)),
  (241, 2): np.array((255, 217, 255)),
  (242, 2): np.array((240, 240, 255)),
  (243, 2): np.array((255, 202, 255)),
  (244, 2): np.array((190, 255, 228)),
  (245, 2): np.array((254, 255, 255)),
  (246, 2): np.array((175, 255, 255)),
  (247, 2): np.array((240, 240, 255)),
  (248, 2): np.array((255, 255, 139)),
  (249, 2): np.array((255, 231, 166)),
  (250, 2): np.array((254, 255, 255)),
  (251, 2): np.array((255, 217, 255)),
  (252, 2): np.array((205, 255, 113)),
  (253, 2): np.array((255, 255, 139)),
  (254, 2): np.array((190, 255, 228)),
  (255, 2): np.array((254, 255, 255)),
  (0, 3): np.array((0, 0, 0)),
  (1, 3): np.array((0, 0, 115)),
  (2, 3): np.array((0, 37, 0)),
  (3, 3): np.array((0, 23, 88)),
  (4, 3): np.array((14, 14, 0)),
  (5, 3): np.array((0, 0, 0)),
  (6, 3): np.array((0, 52, 0)),
  (7, 3): np.array((0, 37, 0)),
  (8, 3): np.array((64, 0, 26)),
  (9, 3): np.array((49, 0, 141)),
  (10, 3): np.array((0, 0, 0)),
  (11, 3): np.array((0, 0, 115)),
  (12, 3): np.array((79, 0, 0)),
  (13, 3): np.array((64, 0, 26)),
  (14, 3): np.array((14, 14, 0)),
  (15, 3): np.array((0, 0, 0)),
  (16, 3): np.array((49, 48, 178)),
  (17, 3): np.array((34, 34, 255)),
  (18, 3): np.array((0, 86, 152)),
  (19, 3): np.array((0, 71, 255)),
  (20, 3): np.array((63, 63, 63)),
  (21, 3): np.array((49, 48, 178)),
  (22, 3): np.array((0, 101, 37)),
  (23, 3): np.array((0, 86, 152)),
  (24, 3): np.array((113, 10, 204)),
  (25, 3): np.array((98, 0, 255)),
  (26, 3): np.array((49, 48, 178)),
  (27, 3): np.array((34, 34, 255)),
  (28, 3): np.array((128, 25, 89)),
  (29, 3): np.array((113, 10, 204)),
  (30, 3): np.array((63, 63, 63)),
  (31, 3): np.array((49, 48, 178)),
  (32, 3): np.array((0, 101, 37)),
  (33, 3): np.array((0, 86, 152)),
  (34, 3): np.array((0, 139, 11)),
  (35, 3): np.array((0, 124, 126)),
  (36, 3): np.array((13, 116, 0)),
  (37, 3): np.array((0, 101, 37)),
  (38, 3): np.array((0, 154, 0)),
  (39, 3): np.array((0, 139, 11)),
  (40, 3): np.array((63, 63, 63)),
  (41, 3): np.array((49, 48, 178)),
  (42, 3): np.array((0, 101, 37)),
  (43, 3): np.array((0, 86, 152)),
  (44, 3): np.array((78, 78, 0)),
  (45, 3): np.array((63, 63, 63)),
  (46, 3): np.array((13, 116, 0)),
  (47, 3): np.array((0, 101, 37)),
  (48, 3): np.array((48, 150, 216)),
  (49, 3): np.array((33, 135, 255)),
  (50, 3): np.array((0, 188, 190)),
  (51, 3): np.array((0, 173, 255)),
  (52, 3): np.array((62, 165, 101)),
  (53, 3): np.array((48, 150, 216)),
  (54, 3): np.array((0, 203, 75)),
  (55, 3): np.array((0, 188, 190)),
  (56, 3): np.array((112, 112, 242)),
  (57, 3): np.array((98, 97, 255)),
  (58, 3): np.array((48, 150, 216)),
  (59, 3): np.array((33, 135, 255)),
  (60, 3): np.array((127, 127, 127)),
  (61, 3): np.array((112, 112, 242)),
  (62, 3): np.array((62, 165, 101)),
  (63, 3): np.array((48, 150, 216)),
  (64, 3): np.array((78, 78, 0)),
  (65, 3): np.array((63, 63, 63)),
  (66, 3): np.array((13, 116, 0)),
  (67, 3): np.array((0, 101, 37)),
  (68, 3): np.array((93, 93, 0)),
  (69, 3): np.array((78, 78, 0)),
  (70, 3): np.array((28, 131, 0)),
  (71, 3): np.array((13, 116, 0)),
  (72, 3): np.array((142, 40, 0)),
  (73, 3): np.array((128, 25, 89)),
  (74, 3): np.array((78, 78, 0)),
  (75, 3): np.array((63, 63, 63)),
  (76, 3): np.array((157, 55, 0)),
  (77, 3): np.array((142, 40, 0)),
  (78, 3): np.array((93, 93, 0)),
  (79, 3): np.array((78, 78, 0)),
  (80, 3): np.array((127, 127, 127)),
  (81, 3): np.array((112, 112, 242)),
  (82, 3): np.array((62, 165, 101)),
  (83, 3): np.array((48, 150, 216)),
  (84, 3): np.array((142, 142, 12)),
  (85, 3): np.array((127, 127, 127)),
  (86, 3): np.array((77, 180, 0)),
  (87, 3): np.array((62, 165, 101)),
  (88, 3): np.array((192, 89, 153)),
  (89, 3): np.array((177, 74, 255)),
  (90, 3): np.array((127, 127, 127)),
  (91, 3): np.array((112, 112, 242)),
  (92, 3): np.array((206, 104, 38)),
  (93, 3): np.array((192, 89, 153)),
  (94, 3): np.array((142, 142, 12)),
  (95, 3): np.array((127, 127, 127)),
  (96, 3): np.array((77, 180, 0)),
  (97, 3): np.array((62, 165, 101)),
  (98, 3): np.array((13, 218, 0)),
  (99, 3): np.array((0, 203, 75)),
  (100, 3): np.array((92, 195, 0)),
  (101, 3): np.array((77, 180, 0)),
  (102, 3): np.array((27, 233, 0)),
  (103, 3): np.array((13, 218, 0)),
  (104, 3): np.array((142, 142, 12)),
  (105, 3): np.array((127, 127, 127)),
  (106, 3): np.array((77, 180, 0)),
  (107, 3): np.array((62, 165, 101)),
  (108, 3): np.array((156, 157, 0)),
  (109, 3): np.array((142, 142, 12)),
  (110, 3): np.array((92, 195, 0)),
  (111, 3): np.array((77, 180, 0)),
  (112, 3): np.array((126, 229, 165)),
  (113, 3): np.array((112, 214, 255)),
  (114, 3): np.array((62, 255, 138)),
  (115, 3): np.array((47, 252, 253)),
  (116, 3): np.array((141, 244, 50)),
  (117, 3): np.array((126, 229, 165)),
  (118, 3): np.array((76, 255, 23)),
  (119, 3): np.array((62, 255, 138)),
  (120, 3): np.array((191, 191, 191)),
  (121, 3): np.array((176, 176, 255)),
  (122, 3): np.array((126, 229, 165)),
  (123, 3): np.array((112, 214, 255)),
  (124, 3): np.array((205, 206, 76)),
  (125, 3): np.array((191, 191, 191)),
  (126, 3): np.array((141, 244, 50)),
  (127, 3): np.array((126, 229, 165)),
  (128, 3): np.array((128, 25, 89)),
  (129, 3): np.array((113, 10, 204)),
  (130, 3): np.array((63, 63, 63)),
  (131, 3): np.array((49, 48, 178)),
  (132, 3): np.array((142, 40, 0)),
  (133, 3): np.array((128, 25, 89)),
  (134, 3): np.array((78, 78, 0)),
  (135, 3): np.array((63, 63, 63)),
  (136, 3): np.array((192, 0, 116)),
  (137, 3): np.array((178, 0, 231)),
  (138, 3): np.array((128, 25, 89)),
  (139, 3): np.array((113, 10, 204)),
  (140, 3): np.array((207, 2, 1)),
  (141, 3): np.array((192, 0, 116)),
  (142, 3): np.array((142, 40, 0)),
  (143, 3): np.array((128, 25, 89)),
  (144, 3): np.array((177, 74, 255)),
  (145, 3): np.array((162, 59, 255)),
  (146, 3): np.array((112, 112, 242)),
  (147, 3): np.array((98, 97, 255)),
  (148, 3): np.array((192, 89, 153)),
  (149, 3): np.array((177, 74, 255)),
  (150, 3): np.array((127, 127, 127)),
  (151, 3): np.array((112, 112, 242)),
  (152, 3): np.array((241, 36, 255)),
  (153, 3): np.array((227, 21, 255)),
  (154, 3): np.array((177, 74, 255)),
  (155, 3): np.array((162, 59, 255)),
  (156, 3): np.array((255, 51, 179)),
  (157, 3): np.array((241, 36, 255)),
  (158, 3): np.array((192, 89, 153)),
  (159, 3): np.array((177, 74, 255)),
  (160, 3): np.array((127, 127, 127)),
  (161, 3): np.array((112, 112, 242)),
  (162, 3): np.array((62, 165, 101)),
  (163, 3): np.array((48, 150, 216)),
  (164, 3): np.array((142, 142, 12)),
  (165, 3): np.array((127, 127, 127)),
  (166, 3): np.array((77, 180, 0)),
  (167, 3): np.array((62, 165, 101)),
  (168, 3): np.array((192, 89, 153)),
  (169, 3): np.array((177, 74, 255)),
  (170, 3): np.array((127, 127, 127)),
  (171, 3): np.array((112, 112, 242)),
  (172, 3): np.array((206, 104, 38)),
  (173, 3): np.array((192, 89, 153)),
  (174, 3): np.array((142, 142, 12)),
  (175, 3): np.array((127, 127, 127)),
  (176, 3): np.array((176, 176, 255)),
  (177, 3): np.array((161, 161, 255)),
  (178, 3): np.array((112, 214, 255)),
  (179, 3): np.array((97, 199, 255)),
  (180, 3): np.array((191, 191, 191)),
  (181, 3): np.array((176, 176, 255)),
  (182, 3): np.array((126, 229, 165)),
  (183, 3): np.array((112, 214, 255)),
  (184, 3): np.array((241, 138, 255)),
  (185, 3): np.array((226, 123, 255)),
  (186, 3): np.array((176, 176, 255)),
  (187, 3): np.array((161, 161, 255)),
  (188, 3): np.array((255, 153, 217)),
  (189, 3): np.array((241, 138, 255)),
  (190, 3): np.array((191, 191, 191)),
  (191, 3): np.array((176, 176, 255)),
  (192, 3): np.array((206, 104, 38)),
  (193, 3): np.array((192, 89, 153)),
  (194, 3): np.array((142, 142, 12)),
  (195, 3): np.array((127, 127, 127)),
  (196, 3): np.array((221, 119, 0)),
  (197, 3): np.array((206, 104, 38)),
  (198, 3): np.array((156, 157, 0)),
  (199, 3): np.array((142, 142, 12)),
  (200, 3): np.array((255, 66, 64)),
  (201, 3): np.array((255, 51, 179)),
  (202, 3): np.array((206, 104, 38)),
  (203, 3): np.array((192, 89, 153)),
  (204, 3): np.array((255, 81, 0)),
  (205, 3): np.array((255, 66, 64)),
  (206, 3): np.array((221, 119, 0)),
  (207, 3): np.array((206, 104, 38)),
  (208, 3): np.array((255, 153, 217)),
  (209, 3): np.array((241, 138, 255)),
  (210, 3): np.array((191, 191, 191)),
  (211, 3): np.array((176, 176, 255)),
  (212, 3): np.array((255, 168, 102)),
  (213, 3): np.array((255, 153, 217)),
  (214, 3): np.array((205, 206, 76)),
  (215, 3): np.array((191, 191, 191)),
  (216, 3): np.array((255, 115, 243)),
  (217, 3): np.array((255, 100, 255)),
  (218, 3): np.array((255, 153, 217)),
  (219, 3): np.array((241, 138, 255)),
  (220, 3): np.array((255, 130, 128)),
  (221, 3): np.array((255, 115, 243)),
  (222, 3): np.array((255, 168, 102)),
  (223, 3): np.array((255, 153, 217)),
  (224, 3): np.array((205, 206, 76)),
  (225, 3): np.array((191, 191, 191)),
  (226, 3): np.array((141, 244, 50)),
  (227, 3): np.array((126, 229, 165)),
  (228, 3): np.array((220, 220, 0)),
  (229, 3): np.array((205, 206, 76)),
  (230, 3): np.array((156, 255, 0)),
  (231, 3): np.array((141, 244, 50)),
  (232, 3): np.array((255, 168, 102)),
  (233, 3): np.array((255, 153, 217)),
  (234, 3): np.array((205, 206, 76)),
  (235, 3): np.array((191, 191, 191)),
  (236, 3): np.array((255, 183, 0)),
  (237, 3): np.array((255, 168, 102)),
  (238, 3): np.array((220, 220, 0)),
  (239, 3): np.array((205, 206, 76)),
  (240, 3): np.array((255, 255, 255)),
  (241, 3): np.array((240, 240, 255)),
  (242, 3): np.array((190, 255, 228)),
  (243, 3): np.array((175, 255, 255)),
  (244, 3): np.array((255, 255, 139)),
  (245, 3): np.array((255, 255, 255)),
  (246, 3): np.array((205, 255, 113)),
  (247, 3): np.array((190, 255, 228)),
  (248, 3): np.array((255, 217, 255)),
  (249, 3): np.array((255, 202, 255)),
  (250, 3): np.array((254, 255, 255)),
  (251, 3): np.array((240, 240, 255)),
  (252, 3): np.array((255, 231, 166)),
  (253, 3): np.array((255, 217, 255)),
  (254, 3): np.array((255, 255, 139)),
  (255, 3): np.array((254, 255, 255)),
}
# 87 unique colours
