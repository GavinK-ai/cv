# import package
import labelme2coco

# set directory that contains labelme annotations and image files
labelme_folder = "C:\SDK\Perantis\Perantis\cv\gavPrj\assets"

# set path for coco json to be saved
save_json_path = "assets\coco.json"

# convert labelme annotations to coco
labelme2coco.convert(labelme_folder, save_json_path)