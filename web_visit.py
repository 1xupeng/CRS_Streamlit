import time
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

def human_like_delay(min_sec=1, max_sec=3):
    """Random delay between actions to simulate human behavior"""
    time.sleep(random.uniform(min_sec, max_sec))

def simulate_mouse_movement(driver, element):
    """Simulate human-like mouse movement to element"""
    action = ActionChains(driver)
    # Move mouse randomly before reaching target
    for _ in range(random.randint(2, 4)):
        x_offset = random.randint(-50, 50)
        y_offset = random.randint(-50, 50)
        action.move_by_offset(x_offset, y_offset)
        human_like_delay(0.1, 0.3)
    action.move_to_element(element)
    action.perform()

def human_like_scroll(driver):
    """Scroll the page in a human-like manner"""
    scroll_pauses = random.randint(3, 6)
    scroll_height = driver.execute_script("return document.body.scrollHeight")
    
    for _ in range(scroll_pauses):
        scroll_amount = random.randint(200, 500)
        current_position = driver.execute_script("return window.pageYOffset")
        
        if current_position + scroll_amount > scroll_height:
            break
            
        driver.execute_script(f"window.scrollBy(0, {scroll_amount})")
        human_like_delay(1, 2)
    
    # Sometimes scroll back up a bit
    if random.random() > 0.7:  # 30% chance
        driver.execute_script("window.scrollBy(0, -200)")
        human_like_delay(1, 2)

def visit_streamlit_app():
    # Set up Chrome options
    options = Options()
    options.add_argument("--headless")  # Run in headless mode
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    # Initialize driver
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    
    try:
        # Open the page with human-like delay
        driver.get("https://crs-mortality-predictor.streamlit.app/")
        print("Page loaded")
        human_like_delay(2, 4)
        
        # Simulate reading the page
        human_like_scroll(driver)
        
        # Find interactive elements (modify selectors for your specific app)
        try:
            # Example: Find buttons/inputs (customize for your app)
            buttons = driver.find_elements(By.CSS_SELECTOR, "button")
            inputs = driver.find_elements(By.CSS_SELECTOR, "input")
            
            # Randomly interact with some elements
            for element in random.sample(buttons + inputs, min(2, len(buttons + inputs))):
                simulate_mouse_movement(driver, element)
                human_like_delay()
                if element.tag_name == "button":
                    element.click()
                    print("Clicked a button")
                    human_like_delay(2, 4)

                    
        except Exception as e:
            print(f"Element interaction skipped: {str(e)}")
        
        # Final scroll and delay before leaving
        human_like_scroll(driver)
        human_like_delay(3, 5)
        
        print("Visit completed successfully")
        
    except Exception as e:
        print(f"Error during visit: {str(e)}")
    finally:
        driver.quit()

if __name__ == "__main__":
    visit_streamlit_app()