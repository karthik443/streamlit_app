import streamlit as st
from PIL import Image, ImageOps
import pillow_heif

# HEIC/HEIF support for iPhone
pillow_heif.register_heif_opener()

from model_inference import load_model, predict

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="ChaatCheck",
    page_icon="🍛",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------
# LOAD MODEL (ONLY ONCE)
# ---------------------------------------------------
@st.cache_resource
def get_model():
    model = load_model("food_model.pth")
    model.eval()
    return model

model = get_model()

# ---------------------------------------------------
# FOOD METADATA
# ---------------------------------------------------
food_data = {
    "aloo_tikki": {
        "calories": "180–220",
        "allergens": ["gluten"],
        "diet": "veg",
        "ingredients": ["potato", "bread crumbs", "spices", "oil"]
    },
    "bhatura": {
        "calories": "300–350",
        "allergens": ["gluten"],
        "diet": "veg",
        "ingredients": ["maida", "yogurt", "oil"]
    },
    "chana_masala": {
        "calories": "220–260",
        "allergens": [],
        "diet": "vegan",
        "ingredients": ["chickpeas", "tomato", "spices"]
    },
    "daal_puri": {
        "calories": "260–300",
        "allergens": ["gluten"],
        "diet": "veg",
        "ingredients": ["wheat flour", "lentils", "oil"]
    },
    "jalebi": {
        "calories": "140–180",
        "allergens": ["gluten"],
        "diet": "veg",
        "ingredients": ["maida", "sugar syrup", "ghee"]
    },
    "kachori": {
        "calories": "250–300",
        "allergens": ["gluten"],
        "diet": "veg",
        "ingredients": ["flour", "lentils", "oil"]
    },
    "lassi": {
        "calories": "150–200",
        "allergens": ["dairy"],
        "diet": "veg",
        "ingredients": ["curd", "sugar", "cardamom"]
    },
    "litti_chokha": {
        "calories": "300–350",
        "allergens": ["gluten"],
        "diet": "veg",
        "ingredients": ["wheat flour", "sattu", "ghee"]
    },
    "poha": {
        "calories": "150–200",
        "allergens": ["nuts"],
        "diet": "vegan",
        "ingredients": ["flattened rice", "peanuts", "spices"]
    },
    "rabri": {
        "calories": "300–400",
        "allergens": ["dairy", "nuts"],
        "diet": "veg",
        "ingredients": ["milk", "sugar", "dry fruits"]
    },
    "kuzhi_paniyaram": {
        "calories": "200–250",
        "allergens": [],
        "diet": "veg",
        "ingredients": ["rice batter", "lentils", "oil"]
    },
    "unni_appam": {
        "calories": "220–260",
        "allergens": ["gluten"],
        "diet": "veg",
        "ingredients": ["rice flour", "banana", "jaggery"]
    }
}

# ---------------------------------------------------
# ALLERGEN BADGES
# ---------------------------------------------------
def show_allergens(allergens):

    if not allergens:
        st.success("✅ No major allergens")
        return

    cols = st.columns(len(allergens))

    for i, allergen in enumerate(allergens):

        if allergen == "gluten":
            cols[i].error("🌾 Gluten")

        elif allergen == "dairy":
            cols[i].warning("🥛 Dairy")

        elif allergen == "nuts":
            cols[i].info("🥜 Nuts")

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.title("🍛 ChaatCheck")
st.caption(
    "Upload or click a street food image → get prediction, calories & allergens. try with <1MB image else will take time to upload"
)


# ---------------------------------------------------
# MOBILE + DESKTOP IMAGE INPUT
# ---------------------------------------------------
tab1, tab2 = st.tabs(["📤 Upload Image", "📸 Use Camera"])

uploaded_file = None

with tab1:

    uploaded_file = st.file_uploader(
        "Upload Food Image",
        type=["jpg", "jpeg", "png", "heic", "heif"]
    )

with tab2:

    camera_file = st.camera_input(
        "Take a Food Picture"
    )

    if camera_file is not None:
        uploaded_file = camera_file

# ---------------------------------------------------
# PROCESS IMAGE
# ---------------------------------------------------
if uploaded_file is not None:

    try:

        # -------------------------
        # OPEN IMAGE
        # -------------------------
        image = Image.open(uploaded_file)

        # -------------------------
        # FIX MOBILE ROTATION
        # -------------------------
        image = ImageOps.exif_transpose(image)

        # -------------------------
        # RGB CONVERSION
        # -------------------------
        image = image.convert("RGB")

        # -------------------------
        # COMPRESS HUGE IMAGES
        # FASTEST IMPORTANT STEP
        # -------------------------
        image.thumbnail((512, 512))

        # -------------------------
        # MODEL INPUT SIZE
        # -------------------------
        model_image = image.resize((224, 224))

        # -------------------------
        # PREDICTION
        # -------------------------
        with st.spinner("🔍 Analyzing food..."):

            pred, conf = predict(model_image, model)

        # ---------------------------------------------------
        # UI LAYOUT
        # ---------------------------------------------------
        col1, col2 = st.columns([1, 1])

        # IMAGE DISPLAY
        with col1:

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

        # RESULT DISPLAY
        with col2:

            st.subheader(
                f"🍽 {pred.replace('_', ' ').title()}"
            )

            st.progress(
                min(int(conf * 100), 100)
            )

            st.write(
                f"### Confidence: {conf * 100:.2f}%"
            )

            info = food_data.get(pred, {})

            if info:

                st.write(
                    f"🔥 Calories: {info['calories']} kcal"
                )

                st.write(
                    f"🥗 Diet: {info['diet'].upper()}"
                )

                st.write("### ⚠ Allergens")

                show_allergens(
                    info["allergens"]
                )

                st.write("### 🧾 Ingredients")

                st.write(
                    ", ".join(info["ingredients"])
                )

    except Exception as e:

        st.error(
            f"❌ Error processing image: {e}"
        )

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")
st.caption("Built with Streamlit + PyTorch")