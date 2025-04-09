# Define profiles (velocities, strain rates, and topography) across different regions of the Alpine-Himalayan belt.
# This organization keeps the main notebook clean and more readable.

profiles = [
    {
        'start_lon': 66.4,
        'start_lat': 42.0,
        'end_lon': 104.9,
        'end_lat': 19.3,
        'width_km': 100,
        'name': 'Tian_Shan_Tibet_SE_Asia_1',
        'annotations': [
            {'x': 450, 'y': 6000, 'text': 'Tian', 'font': "10p,Times-Italic,black"},
            {'x': 450, 'y': 6000, 'text': 'Shan', 'font': "10p,Times-Italic,black", 'offset': "0.0c/-0.3c"},
            {'x': 890, 'y': 6000, 'text': 'Pamir', 'font': "10p,Times-Italic,black"},
            {'x': 1110, 'y': 3500, 'text': 'Tarim', 'font': "8p,Times-Italic,black"},
            {'x': 1110, 'y': 3500, 'text': 'Basin', 'font': "8p,Times-Italic,black", 'offset': "0.0c/-0.3c"},
            {'x': 2100, 'y': 6700, 'text': 'Tibetan Plateau', 'font': "10p,Times-Italic,black"},
            {'x': 2970, 'y': 2000, 'text': 'MFT', 'font': "10p,Times-Italic,black+a90"},
            {'x': 3300, 'y': 3800, 'text': 'Sagaing F.', 'font': "10p,Times-Italic,black+a90"},
        ]
    },
    {
        'start_lon': 64.5,
        'start_lat': 39.6,
        'end_lon': 106.2,
        'end_lat': 22.2,
        'width_km': 100,
        'name': 'Tibet_EW',
        'annotations': [
            {'x': 300, 'y': 4500, 'text': 'Tian Shan', 'font': "9p,Times-Italic,black"},
            {'x': 750, 'y': 6500, 'text': 'Hindu', 'font': "9p,Times-Italic,black"},
            {'x': 750, 'y': 6500, 'text': 'Kush', 'font': "9p,Times-Italic,black", 'offset': "0.0c/-0.3c"},
            {'x': 1070, 'y': 6350, 'text': 'KKF', 'font': "9p,Times-Italic,black+a90"},
            {'x': 1900, 'y': 6800, 'text': 'Tibetan Plateau', 'font': "10p,Times-Italic,black"},
            {'x': 3000, 'y': 5000, 'text': 'MFT', 'font': "10p,Times-Italic,black+a90"},
            {'x': 3300, 'y': 4800, 'text': 'Sagaing F.', 'font': "10p,Times-Italic,black+a90"},
            {'x': 3600, 'y': 4600, 'text': 'Lijiang F.', 'font': "10p,Times-Italic,black+a90"},
            {'x': 3950, 'y': 4300, 'text': 'Xiaojiang F.', 'font': "10p,Times-Italic,black+a90"},
        ],
    },
    {
        'start_lon': 101.1,
        'start_lat': 25.2,
        'end_lon': 104.5,
        'end_lat': 24.4,
        'width_km': 100,
        'name': 'Xianshuihe1',
        'annotations': [
        ],
    },
    {
        'start_lon': 100.8,
        'start_lat': 26.4,
        'end_lon': 104.9,
        'end_lat': 26.1,
        'width_km': 100,
        'name': 'Xianshuihe2',
        'annotations': [
        ],
    },
    {
        'start_lon': 100.1,
        'start_lat': 28.3,
        'end_lon': 105.5,
        'end_lat': 28.3,
        'width_km': 100,
        'name': 'Xianshuihe3',
        'annotations': [
        ],
    },
    {
        'start_lon': 99.9,
        'start_lat': 29.2,
        'end_lon': 104.7,
        'end_lat': 30.1,
        'width_km': 100,
        'name': 'Xianshuihe4',
        'annotations': [
        ],
    },
    {
        'start_lon': 99.6,
        'start_lat': 30,
        'end_lon': 103.4,
        'end_lat': 31.2,
        'width_km': 100,
        'name': 'Xianshuihe5',
        'annotations': [
        ],
    },
]