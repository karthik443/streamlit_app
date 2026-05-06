import streamlit as st
from PIL import Image
from model_inference import load_model, predict

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="ChaatCheck",
    page_icon="🍛",
    layout="centered"
)

# ---------------------------------------------------
# LOAD MODEL (CACHE FOR SPEED)
# ---------------------------------------------------
@st.cache_resource
def get_model():
    model = load_model("food_model.pth")
    model.eval()   # IMPORTANT: inference mode
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
# ALLERGEN DISPLAY
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
st.caption("Upload a street food image → get prediction, calories & allergens")

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Food Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------
if uploaded_file is not None:

    # FAST IMAGE LOADING + OPTIMIZATION
    image = Image.open(uploaded_file)

    # Reduce huge mobile image sizes
    image.thumbnail((512, 512))

    # Convert to RGB
    image = image.convert("RGB")

    # Optional: force resize for faster inference
    # match your training size
    image = image.resize((224, 224))

    # PREDICT
    with st.spinner("🔍 Analyzing food..."):

        pred, conf = predict(image, model)

    # LAYOUT
    col1, col2 = st.columns([1, 1])

    # IMAGE
    with col1:
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    # RESULTS
    with col2:

        st.subheader(f"🍽 {pred.replace('_', ' ').title()}")

        st.progress(min(int(conf * 100), 100))

        st.write(f"### Confidence: {conf * 100:.2f}%")

        info = food_data.get(pred)

        if info:

            st.write(
                f"🔥 Calories: {info['calories']} kcal (per serving)"
            )

            st.write(
                f"🥗 Diet Type: **{info['diet'].upper()}**"
            )

            st.write("### ⚠ Allergens")
            show_allergens(info["allergens"])

            st.write("### 🧾 Ingredients")
            st.write(", ".join(info["ingredients"]))

        else:
            st.warning("No metadata available for this food item.")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")
st.caption("Built with Streamlit + PyTorch")