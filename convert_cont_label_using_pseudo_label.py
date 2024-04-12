import argparse
import json
import os
import glob

hair_tag_cols = [
    'hair_style_name',
    'hair_length',
    'curl_type',
    'curl_width',
    'bangs',
    # 'parting',
    'cut',
    'hair_thickness',
]

label_map = {
    'hair_style_name': ['build perm', 'hippie perm', 'hug perm', 'hush cut', 'layered cut', 'short cut', 'slick cut',
                        'tassel cut'],
    'hair_length': ['bob hair', 'long hair', 'medium hair', 'short hair'],
    'curl_type': ['cs-curl perm', 'inner c-curl perm', 'no-curl', 'outer c-curl perm', 's-curl perm',
                  'twist curl perm'],
    'curl_width': ['thick curl', 'thin curl'],
    'bangs': ['faceline bangs', 'full bangs', 'see-through bangs', 'side bangs'],
    'cut': ['layered hair', 'no-layered hair'],
    'hair_thickness': ['thick hair', 'thin hair'],
}


def main(args):
    output_file = open(args.output_label_path, 'w+', encoding='utf-8')
    human_label_dict = {}
    if args.human_label_path:
        with open(args.human_label_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                fname = os.path.basename(item['source'])
                fname = fname.replace('_exact_hair_mask_00001_', '').replace('_reverse_face_mask_00001_', '')
                human_label_dict[fname] = item['prompt']
    pseudo_label_dict = {}
    for col in hair_tag_cols:
        pseudo_label_dict[col] = json.load(
            open(os.path.join(args.pseudo_label_dir, f'infer_{col}.json'), 'r', encoding='utf-8'))
    with open(args.src_label_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            file_name = os.path.basename(item['source']).split(args.mask_file_name_prefix)[0] + ".jpg"
            ori_file_name = os.path.basename(item['source']).replace('_exact_hair_mask_00001_', '').replace(
                '_reverse_face_mask_00001_', '')
            if human_label_dict and ori_file_name in human_label_dict:
                print("hit human label")
                prompt = human_label_dict[ori_file_name]
            else:
                prompt = item['prompt']
                new_prompt = []
                for col in pseudo_label_dict:
                    label_idx = pseudo_label_dict[col][file_name]['index']
                    score = pseudo_label_dict[col][file_name]['prob']
                    if score < args.score_thr:
                        continue
                    new_prompt.append(label_map[col][label_idx])

                if not new_prompt:
                    continue

                prompt = prompt[prompt.index('1girl'):]
                new_prompt = ', '.join(new_prompt)
                prompt = f"{new_prompt}, {prompt}"
            output_file.write(
                json.dumps(
                    {"source": item["source"], "target": item["target"], "prompt": prompt},
                    ensure_ascii=False)
                + "\n")

    output_file.close()
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_label_path', type=str,
                        default='D:\dataset\hair_controlnet_images_no_filter\images/*.jpg')
    parser.add_argument('--pseudo_label_dir', type=str,
                        default='D:\dataset\hair_controlnet_images_no_filter\images/*.jpg')
    # human_label_path
    parser.add_argument('--human_label_path', type=str, default='./data/human_label')
    parser.add_argument('--output_label_path', type=str, default='./data/output')
    # score_thr
    parser.add_argument('--score_thr', type=float, default=0.7)
    # mask_file_prefix
    parser.add_argument('--mask_file_name_prefix', type=str, default='_reverse_face_mask')

    main(parser.parse_args())
