# Model Input Parameters

This document outlines the 32 input features required by the AWOS LSTM forecasting model, categorized by whether they are raw hardware readings or engineered features.

## 1. Originally Present (Raw Sensor Data)

| Parameter Name | Unit of Measure | Description |
| :--- | :--- | :--- |
| `CLOUD_Cloud_Base_1` | Feet (ft) | Height of the lowest cloud layer |
| `CLOUD_Cloud_Base_2` | Feet (ft) | Height of the second cloud layer |
| `CLOUD_Cloud_Base_3` | Feet (ft) | Height of the third cloud layer |
| `CLOUD_OCTA1` | Oktas (0-8) | Sky cover fraction for the first layer |
| `CLOUD_OCTA2` | Oktas (0-8) | Sky cover fraction for the second layer |
| `CLOUD_OCTA3` | Oktas (0-8) | Sky cover fraction for the third layer |
| `CLOUD_Vertical_Visibility` | Feet (ft) | Vertical visibility distance when sky is obscured |
| `VIS_Visibility_MOR` | Meters (m) | Meteorological Optical Range (horizontal visibility) |
| `VIS_RVR` | Meters (m) | Runway Visual Range |
| `PTU_Pressure_QNH` | Hectopascals (hPa) | Barometric pressure adjusted to sea level |
| `PTU_Temperature` | Degrees Celsius (°C) | Ambient air temperature |
| `PTU_DewPoint` | Degrees Celsius (°C) | Dew point temperature |
| `PTU_Humidity` | Percentage (%) | Relative humidity |
| `PW_Rain_Intensity` | Millimeters/hour (mm/h)| Current rate of precipitation |
| `PW_Rain_Sum` | Millimeters (mm) | Accumulated precipitation |
| `WIND_Wind_Speed` | Knots (kt) | Sustained wind speed |
| `WIND_Wind_Gust` | Knots (kt) | Maximum wind gust speed |

## 2. Created Artificially via Calculations (Engineered Features)

| Parameter Name | Unit of Measure | Description |
| :--- | :--- | :--- |
| `WIND_Wind_Dir_Sin` | Dimensionless (-1 to 1) | Sine projection of wind direction (degrees to radians) |
| `WIND_Wind_Dir_Cos` | Dimensionless (-1 to 1) | Cosine projection of wind direction (degrees to radians) |
| `Hour_Sin` | Dimensionless (-1 to 1) | Sine projection of the decimal hour of the day |
| `Hour_Cos` | Dimensionless (-1 to 1) | Cosine projection of the decimal hour of the day |
| `CLOUD_SENSOR_ON` | Binary (0 or 1) | Cloud sensor hardware heartbeat status |
| `PTU_SENSOR_ON` | Binary (0 or 1) | Pressure/Temp/Humidity sensor hardware heartbeat status |
| `PW_SENSOR_ON` | Binary (0 or 1) | Present weather (rain) sensor hardware heartbeat status |
| `VIS_SENSOR_ON` | Binary (0 or 1) | Visibility sensor hardware heartbeat status |
| `WIND_SENSOR_ON` | Binary (0 or 1) | Anemometer hardware heartbeat status |
| `CLOUD_Cloud_Base_1_IMPUTED` | Binary (0 or 1) | Mask indicating if Base 1 data is real (0) or gap-filled/missing (1) |
| `CLOUD_Cloud_Base_2_IMPUTED` | Binary (0 or 1) | Mask indicating if Base 2 data is real (0) or gap-filled/missing (1) |
| `CLOUD_Cloud_Base_3_IMPUTED` | Binary (0 or 1) | Mask indicating if Base 3 data is real (0) or gap-filled/missing (1) |
| `CLOUD_Vertical_Visibility_IMPUTED`| Binary (0 or 1) | Mask indicating if VV data is real (0) or gap-filled/missing (1) |
| `CLOUD_LONG_GAP` | Binary (0 or 1) | Flag indicating a sustained cloud sensor outage |
| `CLOUD_VERTICAL_LONG_GAP` | Binary (0 or 1) | Flag indicating a sustained vertical visibility outage |