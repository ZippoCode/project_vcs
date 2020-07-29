import os, json

PATH_JSON = '/home/zippo/Scaricati/label_images/completions'
PATH_DEST = '../../yolo/obj/original/with_statues/'


def convert_json(list_json):
    for num, j in enumerate(list_json):
        completions = j['completions'][0]
        name_image = (((j['data'])['image']).split('/')[-1]).split('.')[0] + '.txt'
        file = open(os.path.join(PATH_DEST, name_image), 'w')
        for result in completions['result']:
            bbox = "0 {} {} {} {}\n"
            values = result['value']
            x = float(values['x']) / 100
            y = float(values['y']) / 100
            height = round(float(values['height']) / 100, 6)
            width = round(float(values['width']) / 100, 6)
            x_center = round(x + width / 2, 6)
            y_center = round(y + height / 2, 6)
            bbox = bbox.format(x_center, y_center, width, height)
            file.write(bbox)
        file.close()
        print('Converted file number {}'.format(num))


if __name__ == '__main__':
    jsons = []
    for file in os.listdir(PATH_JSON):
        if os.path.isfile(os.path.join(PATH_JSON, file)) and file.endswith('.json'):
            with open(os.path.join(PATH_JSON, file), 'r') as file:
                jsons.append(json.loads(file.read()))
                file.close()
    print('Converting {} JSON files ...'.format(len(jsons)))
    convert_json(jsons)
