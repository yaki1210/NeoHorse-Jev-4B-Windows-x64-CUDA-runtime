import argparse,json
from runtime import NeoHorseGGUF
ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--image');args=ap.parse_args()
model=NeoHorseGGUF(args.model)
try:
    if args.image:
        from PIL import Image
        image=Image.open(args.image)
        request={'state':'Look at the image.','questions':{'color':{'type':'choice','instructions':'What is the main color?','criteria':{'red':'Mostly red.','blue':'Mostly blue.','green':'Mostly green.'}}}}
    else:
        image=None
        request={'state':'Refund policy: a receipt is required. The customer has no receipt.','questions':{'refund':{'type':'noul','instructions':'Is a refund allowed under the policy?'}}}
    print(json.dumps(model.predict(request,image),indent=2))
finally:model.close()
