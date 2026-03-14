import folium

# Predict risk for recent data
predictions = model.predict(recent_data)
risk_scores = model.predict_proba(recent_data)[:, 1]  # Probability of fire

# Create map
map_australia = folium.Map(location=[-25.2744, 133.7751], zoom_start=4)

# Add high-risk areas as red circles
for idx, row in recent_data.iterrows():
    if risk_scores[idx] > 0.7:
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=10,
            color='red',
            popup=f"Risk: {risk_scores[idx]:.0%}"
        ).add_to(map_australia)

map_australia.save('wildfire_risk_map.html')
