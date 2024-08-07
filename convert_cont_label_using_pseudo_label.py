import argparse
import json
import os
import glob

hair_tag_cols = [
    'hair_style_name',
    'hair_length',
    'curl_type',
    # 'curl_width',
    'bangs',
    # 'parting',
    # 'cut',
    # 'hair_thickness',
]

label_map = {
    'hair_style_name': ['build perm', 'hippie perm', 'hug perm', 'hush cut', 'layered cut', 'short cut', 'slick cut',
                        'tassel cut', 'wave perm'],
    'hair_length': ['bob hair', 'long hair', 'medium hair', 'short hair'],
    'curl_type': ['cs-curl perm', 'inner c-curl perm', 'no-curl', 'outer c-curl perm', 's-curl perm',
                  'twist curl perm', 'wave curl'],
    # 'curl_width': ['thick curl', 'thin curl'],
    'bangs': ['faceline bangs', 'full bangs', 'see-through bangs', 'side bangs'],
    # 'cut': ['layered hair', 'no-layered hair'],
    # 'hair_thickness': ['thick hair', 'thin hair'],
}

ban_hair_attrs = {
    'build perm': ['short hair', 'no-curl', 'twist curl perm', 'wave curl'],
    'hippie perm': ['no-curl', 'cs-curl perm', 'inner c-curl perm', 'outer c-curl perm', 's-curl perm'],
    'hug perm': ['short hair', 'no-curl', 'cs-curl perm', 's-curl perm', 'twist curl perm', 'wave curl',
                 'outer c-curl perm'],
    'hush cut': ['short hair', 'twist curl perm', 'wave curl'],
    'layered cut': ['twist curl perm', 'wave curl'],
    'short cut': ['long hair', 'medium hair', 'bob hair'],
    'slick cut': ['inner c-curl perm', 'outer c-curl perm', 's-curl perm', 'twist curl perm', 'wave curl',
                  'cs-curl perm', 'short hair', 'bob hair'],
    'tassel cut': ['long hair', 'medium hair', 'short hair', 'inner c-curl perm', 'outer c-curl perm', 's-curl perm',
                   'twist curl perm', 'wave curl', 'cs-curl perm'],
    'wave perm': ['no-curl', 'cs-curl perm', 'inner c-curl perm', 'outer c-curl perm', 's-curl perm',],

}

hair_structure = {
    'build perm': {
        'long hair': [
            'cs-curl perm',
            's-curl perm',
        ],
        'medium hair': [
            'cs-curl perm',
        ]
    },
    'hippie perm': {
        'long hair': [
            'twist curl perm',
        ],
        'medium hair': [
            'twist curl perm',
        ],
        'bob hair': [
            'twist curl perm',
        ]
    },
    'hug perm': {
        'long hair': [
            'inner c-curl perm',
        ],
        'medium hair': [
            'inner c-curl perm',
        ],
        'bob hair': [
            'inner c-curl perm',
        ]
    },
    'hush cut': {
        'long hair': [
            'no-curl',
            'cs-curl perm',
            'inner c-curl perm',
        ],
        'medium hair': [
            'no-curl',
            'cs-curl perm',
            'inner c-curl perm',
        ],
        'bob hair': [
            'no-curl',
        ]
    },
    'layered cut': {
        'long hair': [
            'no-curl',
            'cs-curl perm',
            'inner c-curl perm',
            's-curl perm',
        ],
        'medium hair': [
            'no-curl',
            'cs-curl perm',
            'inner c-curl perm',
            'outer c-curl perm',
            's-curl perm',
        ],
        'bob hair': [
            'no-curl',
            'inner c-curl perm',
        ]
    },
    'short cut': {
        'short hair': [
            'no-curl',
            'inner c-curl perm',
            's-curl perm',
        ],
    },
    'slick cut': {
        'long hair': [
            'no-curl',
        ],
        'medium hair': [
            'no-curl',
        ],
    },
    'tassel cut': {
        'bob hair': [
            'no-curl',
        ]
    },
    'wave perm': {
        'long hair': [
            'wave curl',
        ],
        'medium hair': [
            'wave curl',
        ],
        'bob hair': [
            'wave curl',
        ]
    },
    'bob hair': [
        'inner c-curl perm',
        'outer c-curl perm',
    ]
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
        if os.path.exists(os.path.join(args.pseudo_label_dir, f'infer_{col}.json')):
            pseudo_label_dict[col] = json.load(
                open(os.path.join(args.pseudo_label_dir, f'infer_{col}.json'), 'r', encoding='utf-8'))
    num_hit = 0
    result_dict = {}
    with open(args.src_label_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            file_name = os.path.basename(item['source']).split(args.mask_file_name_prefix)[0] + ".jpg"
            ori_file_name = os.path.basename(item['source']).replace('_exact_hair_mask_00001_', '').replace(
                '_reverse_face_mask_00001_', '')
            if human_label_dict and ori_file_name in human_label_dict:
                prompt = human_label_dict[ori_file_name]
                num_hit += 1
            else:
                prompt = item['prompt']
                new_prompt = []
                prompt_dict = {}
                for col in pseudo_label_dict:
                    label_idx = pseudo_label_dict[col][file_name]['index']
                    score = pseudo_label_dict[col][file_name]['prob']
                    if score < args.score_thr:
                        continue
                    prompt_dict[col] = label_map[col][label_idx]
                    new_prompt.append(label_map[col][label_idx])

                if not new_prompt:
                    continue

                if args.use_ban_hair_style_attr and 'hair_style_name' in prompt_dict:
                    hair_style_name = prompt_dict['hair_style_name']
                    for bag_tag in ban_hair_attrs[hair_style_name]:
                        if bag_tag in new_prompt:
                            new_prompt.remove(bag_tag)

                if args.use_hair_structure:
                    str_new_prompt = []
                    if 'hair_style_name' in prompt_dict:
                        str_new_prompt.append(prompt_dict['hair_style_name'])
                        if 'hair_length' in prompt_dict and prompt_dict['hair_style_name'] in hair_structure and \
                                prompt_dict['hair_length'] in hair_structure[prompt_dict['hair_style_name']]:
                            str_new_prompt.append(prompt_dict['hair_length'])
                            if 'curl_type' in prompt_dict and prompt_dict['curl_type'] in \
                                    hair_structure[prompt_dict['hair_style_name']][prompt_dict['hair_length']]:
                                str_new_prompt.append(prompt_dict['curl_type'])
                        else:
                            pass
                    if 'bangs' in prompt_dict:
                        str_new_prompt.append(prompt_dict['bangs'])
                    if not str_new_prompt:
                        continue
                    new_prompt = str_new_prompt

                prompt = prompt[prompt.index('1girl'):]
                new_prompt = ', '.join(new_prompt)
                prompt = f"{new_prompt}, {prompt}"
            result_dict[os.path.splitext(os.path.basename(item["target"]))[0]] = {
                "tags": prompt,
            }
            output_file.write(
                json.dumps(
                    {"source": item["source"], "target": item["target"], "prompt": prompt},
                    ensure_ascii=False)
                + "\n")

    output_file.close()
    json.dump(result_dict, open(args.kohya_output_label_path, 'w+', encoding='utf-8'), ensure_ascii=False, indent=4)
    print(f"hit {num_hit} human label")
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
    parser.add_argument('--kohya_output_label_path', type=str, default='./data/output')
    # score_thr
    parser.add_argument('--score_thr', type=float, default=0.7)
    # mask_file_prefix
    parser.add_argument('--mask_file_name_prefix', type=str, default='_reverse_face_mask')
    # use_hair_structure
    parser.add_argument('--use_hair_structure', action='store_true', default=False)
    # use_ban_hair_style_attr
    parser.add_argument('--use_ban_hair_style_attr', action='store_true', default=False)

    main(parser.parse_args())
