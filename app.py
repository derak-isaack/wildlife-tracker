import torch
import streamlit as st 
import cv2

def load_model(model_path):
    device = torch.device('cpu')
    model = torch.load(model_path, map_location=device)
    return model

#Load the best model from trained weights in local machine  
load_model = load_model('train6/weights/best.pt')

#Define function for detecting animals.
def predict(image):
    with torch.no_grad():
        output = load_model(image)
    return output

#Define main function for deployment   
def main():
    st.title("Image Detection")
    st.write("This is a simple image Detection web application.")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = uploaded_file.read()
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button("Detect"):
            output = predict(image)
            st.write("Predicted class:", output)


if __name__ == '__main__':
    main()
    



    