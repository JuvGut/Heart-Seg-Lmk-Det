import argparse
import sys
sys.path.append("..")
sys.path.append(".")
from detection3d.core.lmk_det_infer import detection


def main():
    long_description = 'Inference engine for 3d medical image segmentation \n' \
                       'It supports multiple kinds of input:\n' \
                       '1. Single image\n' \
                       '2. A text file containing paths of all testing images\n' \
                       '3. A folder containing all testing images\n'

    default_input = '/home/mialab22.team2/CSA/DATA/competition/test_file/competition.csv'
    default_model = '/home/mialab22.team2/CSA/DATA/results/model_augmentation'
    default_output = '/home/mialab22.team2/CSA/DATA/competition/detection_coords'
    default_save_prob = False
    default_gpu_id = 2

    parser = argparse.ArgumentParser(description=long_description)
    parser.add_argument('-i', '--input', default=default_input,
                        help='input folder/file for intensity images')
    parser.add_argument('-m', '--model', default=default_model,
                        help='model root folder')
    parser.add_argument('-o', '--output', default=default_output,
                        help='output folder for segmentation')
    parser.add_argument('-g', '--gpu_id', type=int, default=default_gpu_id,
                        help='the gpu id to run model, set to -1 if using cpu only.')
    parser.add_argument('-s', '--save_prob', type=bool, default=default_save_prob,
                        help='Whether save the probability maps.')

    args = parser.parse_args()
    detection(args.input, args.model, args.gpu_id, False, True, args.save_prob, args.output)


if __name__ == '__main__':
    main()
