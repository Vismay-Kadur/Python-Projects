# Introduction
from datetime import datetime  # Importing date & time

# Defining variables
Input_Mode = str("")
Manaual_Entry_Of_Time = True
current_time = datetime.now()
Hour = float(0)
Minute = float(0)
Hour_Hand_Rotation = float(0)
Minute_Hand_Rotation = float(0)
Angle_Between_Hands = float(0)

# Selecting time input mode
Input_Mode = str(input(
    "Select time input mode. Manual (Enter time manually) or Current (Use current time) \n"))
print(f"input value is {Input_Mode}")

if Input_Mode.lower() == "manual":
    Manaual_Entry_Of_Time = True

    # Manual User input of the time
    print("Enter the time")
    Hour = float(input("Hour: "))
    Minute = float(input("Minute= "))
else:  # Input_Mode.lower() == "current":
    Manaual_Entry_Of_Time = False

    # Automatic input of time
    Hour = float(current_time.hour)
    Minute = float(current_time.minute)

# else:
#     Manaual_Entry_Of_Time = False

# if Manaual_Entry_Of_Time == True:
#     # Manual User input of the time
#     print("Enter the time")
#     Hour = float(input("Hour: "))
#     Minute = float(input("Minute= "))
# else:
#     # Automatic input of time
#     Hour = float(current_time.hour)
#     Minute = float(current_time.minute)

print(f"Hour: {Hour}, Minute: {Minute}")

# Caclculation of the angle between the hands of the clock
if Hour == 12 or Hour == 24:
    Hour_Hand_Rotation = 0
else:
    Hour_Hand_Rotation = Hour*6

if Minute == 60 or Minute == 00:
    Minute_Hand_Rotation = 0
else:
    Minute_Hand_Rotation = Minute*6

Angle_Between_Hands = abs(Hour_Hand_Rotation-Minute_Hand_Rotation)
Minumum_Angle_Between_Hands = min(Angle_Between_Hands, 360-Angle_Between_Hands)

# Displaying the results
print(
    f"Angle between the hands of the clock is {Minumum_Angle_Between_Hands} degrees")
