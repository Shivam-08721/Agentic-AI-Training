import datetime
import random
import re
import requests
import json
import csv
import os

class SimpleAgent:
    """
    A simple agent that parses text input and responds to commands.
    This simulates an AI agent without using any actual AI.
    """
    
    def __init__(self):
        self.reminders = []
        self.name = "SimpleAgent"
        # Default location (will use first city in the list if not found)
        self.default_city = "New York"
        self.default_state = "New York"
        # Load city data from CSV file
        self.cities = self.load_cities()
    
    def load_cities(self):
        """Load cities data from the CSV file"""
        cities_dict = {}
        csv_path = os.path.join(os.path.dirname(__file__), 'top_100_us_cities.csv')
        
        try:
            with open(csv_path, 'r') as file:
                csv_reader = csv.DictReader(file)
                for row in csv_reader:
                    city_key = row['city'].lower()
                    cities_dict[city_key] = {
                        'city': row['city'],
                        'state': row['state'],
                        'latitude': row['latitude'],
                        'longitude': row['longitude']
                    }
            return cities_dict
        except Exception as e:
            print(f"Error loading cities data: {e}")
            # Return an empty dict if file can't be loaded
            return {}
    
    def find_city(self, city_name):
        """Find a city by name in the loaded cities data"""
        city_key = city_name.lower()
        
        # Direct match
        if city_key in self.cities:
            return self.cities[city_key]
        
        # Partial match (e.g. "San" could match "San Francisco")
        for key, city_data in self.cities.items():
            if city_key in key:
                return city_data
        
        # If no match found, use default city
        if self.default_city.lower() in self.cities:
            return self.cities[self.default_city.lower()]
        
        # If default city not found (shouldn't happen), return first city in the list
        if self.cities:
            return list(self.cities.values())[0]
        
        # If no cities data at all, return hardcoded values for New York
        return {
            'city': 'New York',
            'state': 'New York',
            'latitude': '40.7128',
            'longitude': '-74.0060'
        }
    
    def process_input(self, user_input):
        """Process user input and return an appropriate response"""
        user_input = user_input.lower().strip()
        
        # Check for weather-related commands
        if "weather" in user_input:
            # Check if a location is mentioned - this is a simple parser
            location_match = re.search(r"(?:weather|weather.*?)\s+(?:in|for|at)\s+(.+?)(?:\?|$)", user_input)
            if location_match:
                location = location_match.group(1).strip()
                city_data = self.find_city(location)
                return self.get_weather(city_data)
            else:
                # Use default city
                city_data = self.find_city(self.default_city)
                return self.get_weather(city_data)
        
        # Check for time-related commands
        elif "time" in user_input:
            return self.get_time()
        
        # Check for reminder commands
        elif "reminder" in user_input or "remind" in user_input:
            match = re.search(r"remind (?:me )?(?:to )?(.+)", user_input)
            if match:
                reminder = match.group(1)
                return self.set_reminder(reminder)
            else:
                return "What would you like me to remind you about?"
        
        # Check for calculation commands
        elif "calculate" in user_input or any(op in user_input for op in ['+', '-', '*', '/']):
            return self.calculate(user_input)
        
        # Check for greeting
        elif any(greeting in user_input for greeting in ["hello", "hi", "hey"]):
            return f"Hello! I'm {self.name}, your simple agent. I can check the weather, tell time, set reminders, or calculate simple math."
        
        # Default response for unrecognized commands
        else:
            return "I'm sorry, I didn't understand that command. Try asking about weather, time, reminders, or calculations."
    
    def get_weather(self, city_data):
        """Get weather information using the National Weather Service API"""
        city = city_data['city']
        state = city_data['state']
        latitude = city_data['latitude']
        longitude = city_data['longitude']
        
        weather_data = self.get_weather_for_location(latitude, longitude)
        
        if 'error' in weather_data:
            # Return the error directly to demonstrate API failures
            return f"ERROR getting weather for {city}, {state}: {weather_data['error']}"
        
        try:
            # Format the weather information into a human-readable response
            temperature = weather_data['temperature']
            temp_unit = weather_data['temperature_unit']
            forecast = weather_data['forecast']
            wind = weather_data['wind_speed']
            wind_dir = weather_data['wind_direction']
            
            return f"Weather for {city}, {state}: {temperature}°{temp_unit}, {forecast}. Wind: {wind_dir} at {wind}."
        except KeyError as e:
            # Return the specific missing key to demonstrate parsing issues
            return f"ERROR: Could not parse weather data for {city}, {state}. Missing expected field: {str(e)}"
    
    def get_weather_for_location(self, latitude, longitude):
        """
        Get weather forecast for a specific location using the National Weather Service API.
        
        Args:
            latitude (str): Latitude coordinate
            longitude (str): Longitude coordinate
            
        Returns:
            dict: Weather data or error message
        """
        # Define headers with User-Agent (NWS recommends including your app name and contact email)
        headers = {
            'User-Agent': 'SimpleAgent Demo (demo@example.com)',
            'Accept': 'application/geo+json'
        }
        
        try:
            # Step 1: Convert lat/lon to grid location
            point_url = f"https://api.weather.gov/points/{latitude},{longitude}"
            response = requests.get(point_url, headers=headers)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            point_data = response.json()
            
            # Extract forecast URL from the points response
            forecast_url = point_data['properties']['forecast']
            
            # Step 2: Get the actual forecast
            forecast_response = requests.get(forecast_url, headers=headers)
            forecast_response.raise_for_status()
            
            forecast_data = forecast_response.json()
            
            # Extract current period forecast (first period in the list)
            current_forecast = forecast_data['properties']['periods'][0]
            
            # Format the response for our simple agent
            weather_info = {
                'temperature': current_forecast['temperature'],
                'temperature_unit': current_forecast['temperatureUnit'],
                'forecast': current_forecast['shortForecast'],
                'detailed_forecast': current_forecast['detailedForecast'],
                'wind_speed': current_forecast['windSpeed'],
                'wind_direction': current_forecast['windDirection']
            }
            
            return weather_info
            
        except requests.exceptions.HTTPError as http_err:
            return {'error': f"HTTP error: {http_err}"}
        except requests.exceptions.ConnectionError:
            return {'error': "Connection error: Failed to connect to the weather service"}
        except requests.exceptions.Timeout:
            return {'error': "Timeout error: Request timed out"}
        except requests.exceptions.RequestException as req_err:
            return {'error': f"Request error: {req_err}"}
        except (KeyError, ValueError, json.JSONDecodeError) as err:
            return {'error': f"Data processing error: {err}"}
    
    def get_time(self):
        """Get the current time"""
        current_time = datetime.datetime.now().strftime("%H:%M")
        return f"The current time is {current_time}."
    
    def set_reminder(self, reminder_text):
        """Set a reminder"""
        self.reminders.append(reminder_text)
        return f"I've set a reminder for you: {reminder_text}"
    
    def calculate(self, expression):
        """Calculate a simple math expression"""
        # Extract numbers and operators
        try:
            # Remove the word "calculate" if present
            if "calculate" in expression:
                expression = expression.replace("calculate", "")
            
            # Extract math expression using regex
            math_expr = re.search(r'(\d+\s*[\+\-\*/]\s*\d+)', expression)
            if math_expr:
                # Clean up the expression and evaluate it
                expr = math_expr.group(1).replace(" ", "")
                result = eval(expr)  # Note: eval is used for simplicity in this demo
                return f"The result of {expr} is {result}."
            else:
                return "I couldn't find a valid math expression. Try something like 'calculate 5 + 3'."
        except Exception as e:
            return f"Sorry, I couldn't calculate that. Error: {str(e)}"

def main():
    agent = SimpleAgent()
    print(f"Welcome to {agent.name}! Type 'exit' to quit.")
    
    while True:
        user_input = input("> ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        response = agent.process_input(user_input)
        print(response)

if __name__ == "__main__":
    main()
