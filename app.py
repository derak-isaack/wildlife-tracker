# import torch
# import streamlit as st 
# from ultralytics import YOLO 


# device = torch.device('cpu')
# device.load_state_dict(torch.load('wildlife5_vision.pth', map_location=device))
# state_dict = torch.load('wildlife5_vision.pth', map_location=device)
# # print(state_dict.keys())

# # st.header('Wildlife5 Vision')
# # st.write('Wildlife5 Vision')


# def main():
#     st.title('Wildlife5 Vision')
#     st.write('Wildlife5 Vision')
    
#     model = state_dict
#     # Load a model
    
#     model.track(source="elephant-4736008.jpg", show=True, save=True)  # track an image
    
#     st.file_uploader
import torch
import streamlit as st 
from ultralytics import YOLO

device = torch.device('cpu')
device.load_state_dict(torch.load('wildlife5_vision.pth', map_location=device))
state_dict = torch.load('wildlife5_vision.pth', map_location=device)
# Load the model state dictionary


# Instantiate the YOLO model
model = YOLO()

# Load the state dictionary into the model
model_state_dict = model.state_dict()  # Get the model state dictionary

for key in state_dict.keys():
    if 'model.model' in key:
        new_key = key.replace('model.model.', '')  # Remove the redundant 'model.model' prefix
        model_state_dict[new_key] = state_dict[key]
model.load_state_dict(model_state_dict)

def main():
    st.title('Wildlife5 Vision')
    st.write('Wildlife5 Vision')
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Convert the uploaded file to bytes
        image_bytes = uploaded_file.getvalue()
        
        # Perform object detection on the uploaded image
        results = model(image_bytes)
        
        # Display the results
        st.image(results.render(), caption="Detected Objects", use_column_width=True)

if __name__ == "__main__":
    main()

    


if __name__== 'main':
    run(main)

