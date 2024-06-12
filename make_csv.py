"""
실수로 생성된 solution을 csv 파일로 저장하는 코드를 안 넣어서
수동으로 리스트를 넣어서 csv 파일을 생성
(출력으로 나온 리스트 data에 복붙 해주세요)
"""
import pandas as pd

# 리스트 예시
data = [0, 885, 811, 764, 704, 368, 807, 923, 768, 735, 850, 751, 711, 206, 679, 40, 963, 964, 733, 927, 780, 736, 863, 926, 845, 832, 377, 843, 965, 958, 858, 817, 898, 899, 934, 913, 823, 960, 741, 235, 763, 687, 230, 767, 708, 749, 151, 856, 747, 467, 383, 725, 158, 730, 791, 797, 932, 938, 902, 935, 906, 722, 840, 105, 387, 475, 702, 67, 146, 831, 826, 849, 903, 426, 929, 847, 681, 732, 945, 980, 453, 345, 173, 678, 347, 947, 943, 369, 758, 161, 316, 789, 771, 246, 723, 231, 238, 216, 883, 288, 853, 99, 676, 857, 290, 864, 289, 454, 921, 440, 860, 248, 124, 260, 994, 728, 952, 940, 966, 205, 291, 894, 68, 690, 261, 409, 100, 16, 55, 376, 695, 119, 705, 833, 168, 367, 125, 250, 820, 135, 703, 370, 435, 86, 366, 419, 47, 252, 307, 740, 344, 444, 500, 748, 920, 838, 439, 96, 420, 379, 388, 813, 225, 152, 259, 450, 62, 222, 163, 808, 308, 340, 956, 265, 719, 499, 53, 262, 815, 39, 848, 236, 117, 153, 939, 159, 884, 841, 675, 317, 26, 196, 445, 48, 375, 201, 318, 61, 120, 408, 484, 716, 57, 84, 191, 389, 232, 769, 328, 122, 88, 27, 430, 355, 131, 297, 319, 197, 497, 134, 226, 56, 971, 361, 121, 79, 944, 299, 451, 194, 951, 87, 448, 256, 101, 3, 724, 975, 110, 922, 108, 298, 77, 253, 434, 480, 483, 51, 330, 770, 38, 204, 329, 258, 429, 498, 95, 354, 792, 185, 489, 324, 341, 688, 428, 224, 179, 167, 478, 359, 78, 195, 460, 482, 872, 278, 821, 210, 360, 487, 25, 115, 215, 715, 404, 172, 257, 479, 394, 223, 918, 2, 842, 102, 412, 323, 109, 766, 322, 937, 186, 488, 346, 891, 313, 7, 314, 247, 8, 200, 828, 917, 697, 959, 995, 835, 851, 443, 844, 773, 892, 150, 936, 492, 418, 997, 274, 351, 896, 139, 468, 912, 212, 199, 886, 66, 867, 890, 805, 933, 240, 263, 862, 871, 462, 798, 737, 364, 778, 981, 142, 946, 220, 386, 74, 353, 471, 304, 854, 145, 759, 800, 82, 338, 334, 501, 779, 897, 887, 178, 925, 227, 46, 683, 495, 979, 680, 306, 21, 713, 129, 743, 466, 403, 754, 836, 397, 229, 973, 198, 349, 456, 176, 818, 275, 382, 879, 301, 277, 177, 472, 406, 92, 709, 776, 877, 970, 343, 333, 865, 774, 726, 422, 812, 189, 760, 916, 739, 417, 373, 234, 804, 245, 772, 762, 829, 874, 42, 930, 684, 686, 11, 781, 442, 742, 895, 1, 814, 127, 305, 962, 701, 208, 12, 700, 905, 402, 157, 972, 281, 924, 72, 790, 731, 738, 961, 984, 181, 787, 834, 783, 399, 953, 401, 438, 126, 855, 283, 410, 765, 97, 436, 18, 907, 98, 491, 985, 243, 976, 60, 413, 284, 411, 827, 931, 868, 712, 950, 217, 465, 837, 714, 914, 816, 45, 784, 955, 727, 44, 4, 987, 910, 416, 830, 63, 363, 421, 292, 405, 309, 942, 794, 880, 326, 876, 969, 881, 795, 242, 474, 957, 974, 464, 214, 799, 9, 706, 104, 744, 786, 295, 180, 335, 982, 287, 694, 395, 977, 991, 315, 58, 682, 793, 69, 89, 904, 490, 461, 785, 71, 796, 137, 244, 685, 870, 710, 155, 721, 948, 325, 806, 111, 753, 873, 915, 949, 187, 734, 432, 989, 116, 919, 449, 911, 357, 285, 803, 140, 342, 788, 777, 339, 756, 211, 819, 279, 882, 824, 729, 132, 869, 414, 54, 52, 901, 164, 878, 986, 839, 809, 801, 978, 293, 350, 266, 866, 846, 237, 698, 254, 875, 437, 707, 822, 968, 909, 755, 192, 941, 457, 118, 85, 761, 75, 983, 810, 452, 893, 302, 251, 249, 718, 446, 380, 113, 717, 888, 757, 160, 123, 336, 91, 310, 699, 202, 802, 954, 900, 49, 162, 239, 481, 59, 106, 861, 750, 90, 147, 362, 174, 476, 720, 365, 745, 112, 967, 746, 384, 782, 928, 552, 93, 696, 663, 529, 182, 586, 691, 677, 692, 378, 859, 525, 441, 407, 207, 233, 423, 625, 602, 393, 41, 653, 34, 580, 327, 271, 270, 128, 32, 469, 228, 37, 36, 635, 35, 272, 133, 33, 264, 889, 908, 31, 28, 458, 303, 273, 641, 825, 560, 581, 592, 599, 141, 656, 24, 396, 669, 193, 267, 221, 76, 381, 374, 600, 427, 512, 83, 594, 775, 269, 165, 516, 425, 582, 496, 17, 570, 94, 584, 556, 184, 459, 183, 22, 398, 752, 510, 514, 385, 424, 166, 10, 852, 255, 30, 520, 171, 190, 268, 693, 148, 296, 15, 107, 149, 575, 203, 175, 477, 130, 648, 549, 280, 523, 400, 188, 143, 43, 282, 50, 392, 455, 169, 590, 470, 29, 532, 447, 156, 241, 213, 352, 23, 136, 543, 506, 534, 615, 19, 689, 154, 209, 555, 358, 300, 993, 515, 639, 144, 463, 433, 631, 390, 356, 20, 598, 568, 331, 618, 493, 431, 138, 630, 485, 627, 539, 672, 103, 348, 80, 320, 605, 473, 276, 13, 73, 494, 170, 620, 312, 542, 371, 569, 504, 65, 321, 14, 114, 564, 673, 81, 571, 521, 550, 219, 391, 585, 613, 218, 530, 671, 415, 332, 286, 545, 372, 591, 6, 665, 505, 674, 64, 337, 486, 519, 551, 608, 573, 507, 574, 634, 70, 563, 637, 646, 572, 667, 294, 666, 612, 659, 988, 528, 607, 644, 536, 597, 561, 595, 311, 511, 5, 642, 664, 576, 503, 616, 990, 654, 651, 531, 657, 565, 655, 513, 632, 601, 593, 587, 610, 621, 623, 566, 661, 611, 526, 509, 544, 670, 662, 647, 660, 603, 604, 502, 652, 524, 537, 553, 617, 546, 540, 541, 633, 992, 583, 606, 645, 517, 609, 508, 629, 614, 668, 624, 577, 622, 643, 589, 557, 547, 578, 518, 535, 619, 588, 527, 562, 554, 538, 596, 558, 628, 640, 559, 658, 567, 522, 579, 548, 636, 996, 533, 638, 626, 649, 650]

# CSV 파일 경로
csv_file = 'test.csv'

# list를 DataFrame으로 변환
df = pd.DataFrame(data)

# Dataframe을 CSV 파일로 저장
df.to_csv(csv_file, index=False, header=False)
