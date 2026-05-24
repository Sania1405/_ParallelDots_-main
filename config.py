# separate, dedicated file used to store all the settings, constants, and parameters of your software program,easily chnageable and readable which allows separation of logic and data

# --- Model Paths & Names ---
yolomodel_path= "yolov8s-world.pt"
clipmodel_name = "openai/clip-vit-base-patch32"

# --- Thresholds & Hyperparameters ---
YOLO_confthreshold = 0.10
CLIP_confthreshold= 0.20
viusualsimilarity_threshold = 0.75
shelfheight_comparison = 60  # Pixels to group items on the same shelf

# want yolo to detect these objects
yolo_search_words=["bottle","chip bag","snack bag","biscuit pack","biscuit box","carton","juice box","butter bloack","cheese block","tub","yogurt cup"]

clip_totalbrands = [
            # Drinks
            "Coca-Cola", "Pepsi", "Amul", 
            "Sprite", "Fanta", "Mountain Dew", "7UP", "Mirinda",
            "Juice Box", "Energy Drink", "Water Bottle",
            # Snacks
            "Lay's", "Pringles", "Kurkure", "Doritos", "Oreo", 
            "Chips", "Biscuits",
            # Dairy & Others
            "Yogurt", "Nestle", "Yakult", "Mother Dairy", "Hershey's", "Epigamia",
            "Other"
        ]

target_brands = ["Coca-Cola", "Pepsi", "Amul", "Lay's", "Pringles"]
 # These are the ONLY brands we actually care to track for both images rest will push to other category, for now as we have no data trained we are just looking for some specific brands so that if giving more brands AI will not hallucinate