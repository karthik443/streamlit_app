import streamlit as st
from PIL import Image
from model_inference import load_model, predict

# 🔹 Load model (cached for speed)
@st.cache_resource
def get_model():
    return load_model("food_model.pth")

model = get_model()

# 🔹 Metadata (same as yours)
food_data = {
    "aloo_tikki": {"calories": "180–220", "allergens": ["gluten"], "diet": "veg", "ingredients": ["potato", "bread crumbs", "spices", "oil"]},
    "bhatura": {"calories": "300–350", "allergens": ["gluten"], "diet": "veg", "ingredients": ["maida", "yogurt", "oil"]},
    "chana_masala": {"calories": "220–260", "allergens": [], "diet": "vegan", "ingredients": ["chickpeas", "tomato", "spices"]},
    "daal_puri": {"calories": "260–300", "allergens": ["gluten"], "diet": "veg", "ingredients": ["wheat flour", "lentils", "oil"]},
    "jalebi": {"calories": "140–180", "allergens": ["gluten"], "diet": "veg", "ingredients": ["maida", "sugar syrup", "ghee"]},
    "kachori": {"calories": "250–300", "allergens": ["gluten"], "diet": "veg", "ingredients": ["flour", "lentils", "oil"]},
    "lassi": {"calories": "150–200", "allergens": ["dairy"], "diet": "veg", "ingredients": ["curd", "sugar", "cardamom"]},
    "litti_chokha": {"calories": "300–350", "allergens": ["gluten"], "diet": "veg", "ingredients": ["wheat flour", "sattu", "ghee"]},
    "poha": {"calories": "150–200", "allergens": ["nuts"], "diet": "vegan", "ingredients": ["flattened rice", "peanuts", "spices"]},
    "rabri": {"calories": "300–400", "allergens": ["dairy", "nuts"], "diet": "veg", "ingredients": ["milk", "sugar", "dry fruits"]},
    "kuzhi_paniyaram": {"calories": "200–250", "allergens": [], "diet": "veg", "ingredients": ["rice batter", "lentils", "oil"]},
    "unni_appam": {"calories": "220–260", "allergens": ["gluten"], "diet": "veg", "ingredients": ["rice flour", "banana", "jaggery"]}
}

# 🔹 Allergen badge function
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

# 🔹 UI
st.set_page_config(page_title="ChaatCheck", layout="centered")

st.title("🍛ChaatCheck- street food recognizer")
st.caption("Upload a food image → get dish, calories & allergens")

file = st.file_uploader("📤 Upload Food Image", type=["jpg", "png", "jpeg"])

if file:
    image = Image.open(file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        pred, conf = predict(image, model)

        st.subheader(f"🍽 {pred.replace('_', ' ').title()}")
        st.progress(int(conf * 100))
        st.write(f"Confidence: {conf*100:.2f}%")

        info = food_data.get(pred, {})

        if info:
            st.write(f"🔥 Calories: {info['calories']} kcal (per serving)")
            st.write(f"🥗 Diet: {info['diet'].upper()}")

            st.write("⚠ Allergens")
            show_allergens(info["allergens"])

            st.write("🧾 Ingredients")
            st.write(", ".join(info["ingredients"]))