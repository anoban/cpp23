#include <cstdlib>
#include <iostream>

#include <intrin.h>
// unlike LLVM, MSVC offers the SSSE3 intrinsic _mm_shuffle_pi8 only in x86 mode

#include <gtest/gtest.h>

namespace endian {

    static constexpr unsigned short __stdcall ushort_from_be_bytes(_In_reads_bytes_(2) const unsigned char* const bytestream) noexcept {
        return static_cast<unsigned short>(bytestream[0]) << 8 | bytestream[1]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    }

    static unsigned long __stdcall ulong_from_be_bytes(_In_reads_bytes_(4) const unsigned char* const bytestream) noexcept {
#if defined(__llvm__) && defined(__clang__)
        static constexpr __m64 mask_pi8 { 0x0405060700010203 };
        return ::_mm_shuffle_pi8(
            *reinterpret_cast<const __m64*>(bytestream), mask_pi8
        )[0]; // LLVM defines __m64 as a vector of 1 long long, hence the array subscript at the end
#elif defined(_MSC_VER) && defined(_MSC_FULL_VER)
        const __m128i operand_epi8 {
            .m128i_u8 {
                       bytestream[0], bytestream[1], bytestream[2], bytestream[3], /* 12 filler zeroes */ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
        };
        static constexpr __m128i mask_epi8 {
            .m128i_u8 { 3, 2, 1, 0, /* we don't care about the rest */ 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }
        };

        return ::_mm_shuffle_epi8(operand_epi8, mask_epi8).m128i_u32[0]; // MSVC defines __m128i as a union
#endif
    }

    static unsigned long long __stdcall ullong_from_be_bytes(_In_reads_bytes_(8) const unsigned char* const bytestream) noexcept {
        static constexpr __m64 mask_pi8 { 0x01020304050607 }; // 0b00000000'00000001'00000010'00000011'00000100'00000101'00000110'00000111
#if defined(__llvm__) && defined(__clang__)
        return ::_mm_shuffle_pi8(*reinterpret_cast<const __m64*>(bytestream), mask_pi8)[0];
#elif defined(_MSC_VER) && defined(_MSC_FULL_VER)
        const __m128i operand_epi8 {
            .m128i_u8 { bytestream[0],
                       bytestream[1],
                       bytestream[2],
                       bytestream[3],
                       bytestream[4],
                       bytestream[5],
                       bytestream[6],
                       bytestream[7],
                       0, /* 8 filler zeroes */
                        0, 0,
                       0, 0,
                       0, 0,
                       0 }
        };
        static constexpr __m128i mask_epi8 {
            .m128i_u8 { 7, 6, 5, 4, 3, 2, 1, 0, /* we don't care about the rest */ 8, 9, 10, 11, 12, 13, 14, 15 }
        };

        return ::_mm_shuffle_epi8(operand_epi8, mask_epi8).m128i_u64[0];
#endif
    }

} // namespace endian

constexpr unsigned short full { 0b11111111'11110000 };
static_assert(full >> 1 == 0b01111111'11111000);

static constexpr unsigned char bytes[] {
    73,  78,  147, 76,  52,  216, 93,  204, 54,  189, 210, 86,  233, 246, 134, 37,  110, 160, 80,  74,  198, 108, 190, 232, 112, 250, 132,
    130, 217, 49,  194, 55,  181, 228, 240, 166, 193, 135, 156, 38,  0,   105, 125, 109, 137, 68,  235, 236, 201, 18,  184, 107, 169, 112,
    11,  125, 157, 95,  41,  171, 9,   154, 236, 170, 85,  119, 87,  146, 172, 230, 41,  49,  240, 141, 248, 161, 20,  80,  26,  126, 90,
    172, 106, 84,  238, 134, 233, 196, 173, 38,  90,  227, 116, 196, 185, 152, 98,  194, 15,  110, 197, 223, 24,  209, 128, 248, 211, 32,
    225, 68,  210, 172, 107, 98,  190, 76,  245, 81,  43,  241, 23,  20,  181, 125, 200, 133, 93,  84,  213, 88,  168, 122, 37,  218, 119,
    227, 55,  194, 222, 95,  190, 251, 116, 133, 216, 174, 142, 138, 171, 40,  96,  102, 51,  76,  2,   3,   139, 81,  12,  91,  126, 174,
    242, 202, 54,  59,  77,  171, 117, 119, 213, 23,  98,  1,   229, 123, 165, 44,  29,  169, 52,  103, 4,   153, 224, 112, 152, 11,  150,
    248, 107, 246, 205, 57,  35,  69,  41,  96,  138, 125, 16,  171, 14,  101, 244, 53,  94,  64,  52,  22,  247, 83,  148, 54,  254, 219,
    244, 160, 139, 162, 151, 150, 223, 14,  94,  216, 22,  41,  66,  228, 136, 21,  39,  209, 189, 148, 185, 203, 61,  166, 168, 133, 17,
    141, 242, 127, 216, 65,  29,  236, 5,   153, 221, 57,  35,  129, 39,  88,  148, 229, 126, 133, 121, 6,   132, 124, 235, 37,  109, 192,
    236, 122, 166, 95,  159, 42,  103, 13,  153, 115, 170, 58,  182, 59,  87,  147, 96,  170, 75,  47,  109, 192, 48,  182, 189, 40,  197,
    191, 250, 239, 173, 248, 169, 124, 120, 138, 147, 124, 245, 194, 8,   200, 71,  150, 213, 225, 55,  196, 124, 82,  231, 243, 0,   75,
    185, 185, 240, 214, 218, 139, 37,  133, 208, 61,  116, 198, 214, 161, 115, 248, 170, 57,  65,  113, 104, 144, 197, 146, 127, 57,  46,
    107, 111, 34,  29,  231, 119, 133, 84,  91,  178, 4,   140, 41,  137, 2,   153, 156, 114, 181, 24,  224, 18,  179, 245, 205, 22,  132,
    133, 217, 160, 55,  166, 75,  170, 123, 11,  130, 13,  203, 230, 147, 99,  179, 53,  208, 162, 233, 49,  208
};

static constexpr unsigned short usorts[] {
    18766, 20115, 37708, 19508, 13528, 55389, 24012, 52278, 14013, 48594, 53846, 22249, 59894, 63110, 34341, 9582,  28320, 41040, 20554,
    19142, 50796, 27838, 48872, 59504, 28922, 64132, 33922, 33497, 55601, 12738, 49719, 14261, 46564, 58608, 61606, 42689, 49543, 34716,
    39974, 9728,  105,   27005, 32109, 28041, 35140, 17643, 60396, 60617, 51474, 4792,  47211, 27561, 43376, 28683, 2941,  32157, 40287,
    24361, 10667, 43785, 2458,  39660, 60586, 43605, 21879, 30551, 22418, 37548, 44262, 58921, 10545, 12784, 61581, 36344, 63649, 41236,
    5200,  20506, 6782,  32346, 23212, 44138, 27220, 21742, 61062, 34537, 59844, 50349, 44326, 9818,  23267, 58228, 29892, 50361, 47512,
    39010, 25282, 49679, 3950,  28357, 50655, 57112, 6353,  53632, 33016, 63699, 54048, 8417,  57668, 17618, 53932, 44139, 27490, 25278,
    48716, 19701, 62801, 20779, 11249, 61719, 5908,  5301,  46461, 32200, 51333, 34141, 23892, 21717, 54616, 22696, 43130, 31269, 9690,
    55927, 30691, 58167, 14274, 49886, 56927, 24510, 48891, 64372, 29829, 34264, 55470, 44686, 36490, 35499, 43816, 10336, 24678, 26163,
    13132, 19458, 515,   907,   35665, 20748, 3163,  23422, 32430, 44786, 62154, 51766, 13883, 15181, 19883, 43893, 30071, 30677, 54551,
    5986,  25089, 485,   58747, 31653, 42284, 11293, 7593,  43316, 13415, 26372, 1177,  39392, 57456, 28824, 38923, 2966,  38648, 63595,
    27638, 63181, 52537, 14627, 9029,  17705, 10592, 24714, 35453, 32016, 4267,  43790, 3685,  26100, 62517, 13662, 24128, 16436, 13334,
    5879,  63315, 21396, 37942, 14078, 65243, 56308, 62624, 41099, 35746, 41623, 38806, 38623, 57102, 3678,  24280, 55318, 5673,  10562,
    17124, 58504, 34837, 5415,  10193, 53693, 48532, 38073, 47563, 52029, 15782, 42664, 43141, 34065, 4493,  36338, 62079, 32728, 55361,
    16669, 7660,  60421, 1433,  39389, 56633, 14627, 9089,  33063, 10072, 22676, 38117, 58750, 32389, 34169, 30982, 1668,  33916, 31979,
    60197, 9581,  28096, 49388, 60538, 31398, 42591, 24479, 40746, 10855, 26381, 3481,  39283, 29610, 43578, 15030, 46651, 15191, 22419,
    37728, 24746, 43595, 19247, 12141, 28096, 49200, 12470, 46781, 48424, 10437, 50623, 49146, 64239, 61357, 44536, 63657, 43388, 31864,
    30858, 35475, 37756, 31989, 62914, 49672, 2248,  51271, 18326, 38613, 54753, 57655, 14276, 50300, 31826, 21223, 59379, 62208, 75,
    19385, 47545, 47600, 61654, 55002, 55947, 35621, 9605,  34256, 53309, 15732, 29894, 50902, 54945, 41331, 29688, 63658, 43577, 14657,
    16753, 29032, 26768, 37061, 50578, 37503, 32569, 14638, 11883, 27503, 28450, 8733,  7655,  59255, 30597, 34132, 21595, 23474, 45572,
    1164,  35881, 10633, 35074, 665,   39324, 40050, 29365, 46360, 6368,  57362, 4787,  46069, 62925, 52502, 5764,  33925, 34265, 55712,
    41015, 14246, 42571, 19370, 43643, 31499, 2946,  33293, 3531,  52198, 59027, 37731, 25523, 45877, 13776, 53410, 41705, 59697, 12752
};

static constexpr unsigned ulongs[] {
    1229886284, 1318276148, 2471245016, 1278531677, 886595020,  3630025782, 1573664445, 3426139602, 918409814,  3184678633, 3528911350,
    1458173574, 3925247525, 4135986542, 2250600096, 628006992,  1856000074, 2689616582, 1347077740, 1254517950, 3329015528, 1824450672,
    3202904314, 3899718276, 1895466114, 4202988249, 2223167793, 2195272130, 3643916855, 834811829,  3258430948, 934667504,  3051679910,
    3840976577, 4037460359, 2797701020, 3246890022, 2275157504, 2619736169, 637561213,  6913389,    1769827721, 2104330564, 1837712619,
    2302995436, 1156312265, 3958163730, 3972600504, 3373447275, 314076073,  3094063472, 1806266379, 2842692477, 1879801245, 192781663,
    2107465513, 2640259499, 1596566281, 699074970,  2869533420, 161148074,  2599201365, 3970585975, 2857727831, 1433884562, 2002227884,
    1469230310, 2460804649, 2900764977, 3861459440, 691138701,  837848568,  4035836065, 2381881620, 4171306064, 2702463002, 340793982,
    1343913562, 444488364,  2119871594, 1521248852, 2892649710, 1783950982, 1424918249, 4001819076, 2263467181, 3921980710, 3299681882,
    2904972003, 643490676,  1524856004, 3816080569, 1959049624, 3300497506, 3113771714, 2556609039, 1656885102, 3255791301, 258917855,
    1858461464, 3319732433, 3742945664, 416383224,  3514890451, 2163790624, 4174586081, 3542147396, 551634130,  3779383980, 1154657387,
    3534515042, 2892718782, 1801633356, 1656638709, 3192714577, 1291145515, 4115737585, 1361834263, 737220372,  4044821685, 387233149,
    347438536,  3044919429, 2110293341, 3364183380, 2237486293, 1565840728, 1423268008, 3579357306, 1487436325, 2826577370, 2049301111,
    635074531,  3665290039, 2011379650, 3812082398, 935517791,  3269353406, 3730816763, 1606351732, 3204150405, 4218717656, 1954928814,
    2245570190, 3635318410, 2928577195, 2391452456, 2326472800, 2871550054, 677406259,  1617310540, 1714637826, 860619267,  1275200395,
    33786705,   59461900,   2337344603, 1359764350, 207322798,  1535028978, 2125394634, 2935147062, 4073338427, 3392551757, 909856171,
    994945909,  1303082359, 2876602325, 1970787607, 2010453858, 3575079425, 392298981,  1644291451, 31816613,   3850085676, 2074422301,
    2771131817, 740141364,  497628263,  2838783748, 879166617,  1728354784, 77193328,   2581622936, 3765475339, 1889012630, 2550896376,
    194443371,  2532862966, 4167825101, 1811336505, 4140644643, 3443073861, 958612777,  591735136,  1160339594, 694192765,  1619688720,
    2323452075, 2098244366, 279645797,  2869847540, 241562677,  1710503262, 4097138240, 895369268,  1581265942, 1077155575, 873920339,
    385307540,  4149449782, 1402222334, 2486632155, 922672116,  4275827872, 3690242187, 4104162210, 2693505687, 2342688662, 2727843551,
    2543247118, 2531200606, 3742260952, 241096726,  1591219753, 3625331010, 371802852,  692249736,  1122273301, 3834123559, 2283087825,
    354931133,  668056980,  3518862521, 3180640715, 2495204157, 3117104550, 3409815208, 1034332293, 2796061969, 2827293069, 2232520178,
    294515327,  2381479896, 4068464705, 2144878877, 3628146156, 1092480005, 502007193,  3959790045, 93969721,   2581412131, 3711509377,
    958628135,  595666776,  2166839444, 660116709,  1486153086, 2498068101, 3850274169, 2122676486, 2239301252, 2030470268, 109346027,
    2222779173, 2095785325, 3945098688, 627949804,  1841359994, 3236723366, 3967460959, 2057723807, 2791284522, 1604266599, 2670356237,
    711396761,  1728944499, 228160426,  2574494266, 1940535990, 2855974459, 985021271,  3057342355, 995595104,  1469276330, 2472585803,
    1621773103, 2857054061, 1261399488, 795721776,  1841311926, 3224417981, 817282344,  3065850053, 3173565887, 684048378,  3317693167,
    3220893613, 4210011640, 4021156009, 2918754684, 4171857016, 2843506826, 2088274579, 2022347644, 2324921589, 2474440130, 2096480776,
    4123134152, 3255355463, 147343254,  3360134869, 1201067489, 2530599223, 3588306884, 3778528380, 935623762,  3296482023, 2085808115,
    1390932736, 3891462219, 4076882873, 4962745,    1270462960, 3115970774, 3119568602, 4040612491, 3604646693, 3666552197, 2334492112,
    629526589,  2245016948, 3493688518, 1031063254, 1959188129, 3335954803, 3600905208, 2708732074, 1945676345, 4171905345, 2855879025,
    960590184,  1097951376, 1902678213, 1754318226, 2428867199, 3314712377, 2457811246, 2134453867, 959343471,  778792738,  1802445341,
    1864506855, 572385143,  501708677,  3883369812, 2005226587, 2236898226, 1415295492, 1538393228, 2986642473, 76294537,   2351532290,
    696844953,  2298648988, 43621490,   2577167029, 2624763160, 1924471008, 3038306322, 417338035,  3759322101, 313783757,  3019230486,
    4123858564, 3440804997, 377783769,  2223364512, 2245632055, 3651155878, 2688001611, 933645226,  2789976699, 1269463819, 2860190594,
    2064351757, 193072587,  2181942246, 231466643,  3420885859, 3868418995, 2472784693, 1672689104, 3006648482, 902865641,  3500337457
};

static constexpr unsigned long long ullongs[] {
    5282321368465563084,  5661952946386881590,  10613916525696661181, 5491251743041174994,  3807896616614875734,  15590842020511504105,
    6758837329681902070,  14715157543578629766, 3944540119380690469,  13678090581140972910, 15156558840983809696, 6262807812849442896,
    16858809750435942474, 17763926937275747014, 9666253810041538156,  2697269493553851582,  7971459622532595432,  11551815260293752944,
    5785654841672495354,  5388113571394681476,  14298012822531638402, 7835955973608211161,  13756369283070482737, 16749162461228773826,
    8140964973950124599,  18051697075762116533, 9548432967713928676,  9428622005104927984,  15650503724619853990, 3585489507709920961,
    13994854361971736967, 4014366365111650204,  13106865414557113382, 16496868785192183296, 17340760203421155433, 12016034385523403133,
    13945286458205633901, 9771727074698816905,  11251681172107659588, 2738304560870802667,  29692781962521580,    7601352182405524681,
    9038030956311398674,  7892915602024108728,  9891290083830708331,  4966323362452761513,  17000183775657437552, 17062189246559383563,
    14488845723748010877, 1348946463870909853,  13288901424180993375, 7757855027776806697,  12209271223940491691, 8073684871851649801,
    827990938552568218,   9051495458652396268,  11339828201319492778, 6857199965390547541,  3002504137572767095,  12324552196536760151,
    692125709077272466,   11163484860395786924, 17053536910050503910, 12273847577474819625, 6158487302930049329,  8599503284780741104,
    6310296132433080461,  10569075490137607672, 12458690713633028257, 16584842012012355860, 2968418121966228560,  3598532201262895130,
    17333783911533124222, 10230103662187413082, 17915623126930971308, 11606990214359854186, 1463699008884861524,  5772064800333518062,
    1909062988616494726,  9104779169374308073,  6533714072419363268,  12423835905497351341, 7662011129279065382,  6119977282228266586,
    17187682058833910499, 9721517518607803252,  16844778886517716164, 14172025774209811641, 12476759750639663512, 2763771412001429602,
    6549206671403016898,  16389941245312680463, 8414054067977981806,  14175528852055355077, 13373547679098783199, 10980552213021450008,
    7116267329639356625,  13983517164139237760, 1112043719991853304,  7982031212271171795,  14258142233369301792, 16075829221759590625,
    1788352333225189700,  15096339536619324626, 9293409969250816684,  17929710693386464363, 15213407226966076258, 2369250550600131262,
    16232330594927951436, 4959215716906454261,  15180626515802781009, 12424132566506098987, 7737956347518462961,  7115209077804495127,
    13712604694414694164, 5545427765346899125,  17676958326880253309, 5849033622504701384,  3166337390729873541,  17372376857336907101,
    1663153714246278484,  1492237151727604949,  13077829368075834712, 9063640885984843944,  14449057598426097786, 9609930455170710053,
    6725234720331408858,  6112889549852367479,  15373222570603739107, 6388490374422717239,  12140057365775071170, 8801681255213548254,
    2727624342103055967,  15742300851128917950, 8638809820320743163,  16372769230673607540, 4018018320375313541,  14041765962054927832,
    16023735986408511662, 6899228157058526862,  13761721204575473290, 18119254366506355371, 8396355324529519400,  9644650529248979040,
    15613573684368269414, 12578143277013820979, 10271210090076189516, 9992124592748186626,  12333213571617653251, 2909437729785906059,
    6946295876809886545,  7364313387214000396,  3696331608409836635,  5476943993731046270,  145112793221922478,   255386917393051378,
    10038818631492498122, 5840143416451844662,  890444641198552635,   6592899262314855245,  9128500445033745835,  12606360641235430261,
    17494855330808165751, 14570898849178941397, 3907802500479571223,  4273260142454445922,  5596696119474610689,  12354912909864862181,
    8464468321071392123,  8634833570258844581,  15354849214827570476, 1684911295723547677,  7062178009908518313,  136651313044629812,
    16535992065715680359, 8909575943726851844,  11901920527799223449, 3178882954525186528,  2137297115227480176,  12192483360657928344,
    3775991871515432971,  7423227275054156694,  331542821780297464,   11087986080917944427, 16172593437432376310, 8113247471748773581,
    10956016512216255801, 835127923509639459,   10878563607662633797, 17900672505201509673, 7779631051617675616,  17783933327202934922,
    14787889631401642621, 4117210528362429712,  2541483059337564331,  4983620610582162190,  2981535223074459237,  6956510084969948660,
    9979150676189901877,  9011890932696757598,  1201069556675993152,  12325901329701421108, 1037503799230477334,  7346555571068475127,
    17597074748864919379, 3845581724288766868,  6791485511318082614,  4626347968731297534,  3753459277800865499,  1654883284124883956,
    17821751114360157344, 6022499069941031051,  10680003787011165090, 3962846565844624023,  18364540875907962774, 15849469510212359903,
    17627242471972331278, 11568518839786212958, 10061771191742258904, 11715998840390604822, 10923163199047472681, 10871423826010712386,
    16072888402309628644, 1035502554034922632,  6834236801006471189,  15570678128958772519, 1596881092182616017,  2973189977139564989,
    4820127125637021076,  16467435298246988985, 9805787545451411915,  1524417611062430525,  2869282884081630630,  15113399450224928424,
    13660747852285388933, 10716820253954311441, 13387862103290089869, 14645044805995957746, 4442423371926205055,  12008994716825845720,
    12143131271630936129, 9588601154314977565,  1264933701263891948,  10228378270493961221, 17473922853407294873, 9212184634555996637,
    15582769085222083897, 4692165895590328611,  2156104480003269505,  17007168743259996455, 403596879104911192,   11087080682309507220,
    15940811393672451301, 4117276490336626046,  2558369324731825797,  9306504551513097593,  2835179678820825350,  6382978903458776708,
    10729120799006295164, 16536801636597923051, 9116826089580981029,  9617725645327639917,  8720803400505454016,  469637610540482796,
    9546763856106286202,  9001329433548454566,  16944069848419968607, 2697023873767333791,  7908580957184040746,  13901621004773304935,
    17040115069732153101, 8837856455977012633,  11988475737549937011, 6890272576998306730,  11469092709159119418, 3055425824915864246,
    7425760082660079163,  979941568896449367,   11057368679266867091, 8334538614756578144,  12266316901085569194, 4230634147281939019,
    13131185429022395183, 4276048414594772845,  6310493787398303168,  10619675161234620464, 6965462440758751414,  12270953758123407037,
    5417669548968426792,  3417599007700887749,  7908374507078337983,  13848769777713397754, 3510200942395914991,  13167725715295760301,
    13630361700576243192, 2937965416413001897,  14249383653346421116, 13833632735902137464, 18081862312422832266, 17270733552857156243,
    12535955914849162108, 17917989449633070325, 12212768826097202626, 8969071023969649160,  8685916996245784776,  9985462193774708807,
    10627639434407331734, 9004316372972836565,  17708726341061760481, 13981645252970537271, 632834460804528068,   14431669376282772604,
    5158545586479463506,  10868840905364493031, 15411660716877473779, 16228655820498793216, 4018473463041949771,  14158282484713802681,
    8958477639661369785,  5974010613326264816,  16713702967340560598, 17510078612277090010, 21314831514000011,    5456596867584002853,
    13382992573288359301, 13398445125552932304, 17354298505283620925, 15481839662314569076, 15747721778685637830, 10026567274841032406,
    2703796113676621473,  9642274373961687411,  15005277930821612536, 4428382958746073258,  8414648942712105529,  14327816783991028033,
    15465770107211956593, 11633915672416842088, 8356616271473764496,  17918197020685275333, 12265907015461684626, 4125703427567489663,
    4715665255832911673,  8171940702104533294,  7534739409581190763,  10431905186991467375, 14236581255640215330, 10556218923113456157,
    9167409555450240487,  4120348834148509559,  3344889340574005125,  7741443796305937748,  8007995967398040667,  2458375472138181554,
    2154822361249722884,  16678946342352061580, 8612382615221341225,  9607404725226711433,  6078647854667761922,  6607348603344716441,
    12827531748678211996, 327682541322083442,   10099754283615154869, 2992926286142420248,  9872622230367967456,  187352875991097362,
    11068848106301821619, 11273271935704937461, 8265540041773938125,  13049426291239275798, 1792453215825761924,  16146165482365813893,
    1347690974708794841,  12967496198679550368, 17711837667955154999, 14778144937679533990, 1622568935502620235,  9549277867060644778,
    9644916237864249979,  15681595089877629707, 11544879013700504450, 4009975713800880653,  11982858679000108491, 5452305588242205670,
    12284425061788280467, 8866323287176024931,  829240450787533747,   9371370590803571509,  994141663472596432,   14692592890760515746,
    16614733071653053161, 10620529389984737585
};

TEST(ENDIAN, TO_USHORT) {
    for (size_t i { 0 }; i < __crt_countof(bytes) - 1; ++i) EXPECT_EQ(endian::ushort_from_be_bytes(bytes + i), usorts[i]);
}

TEST(ENDIAN, TO_ULONG) {
    for (size_t i { 0 }; i < __crt_countof(bytes) - 4; ++i) EXPECT_EQ(endian::ulong_from_be_bytes(bytes + i), ulongs[i]);
}

TEST(ENDIAN, TO_ULLONG) {
    for (size_t i { 0 }; i < __crt_countof(bytes) - 8; ++i) EXPECT_EQ(endian::ullong_from_be_bytes(bytes + i), ullongs[i]);
}

auto wmain() -> int {
    testing::InitGoogleTest();
    return ::RUN_ALL_TESTS();
}