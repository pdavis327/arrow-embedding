# 1. Countries Dictionary: Defines each target country along with lists of adjectives, component types, and specifications commonly associated with their electronics manufacturing. This helps inject subtle "signals" into the descriptions.
# 2. Random Selection: For each sample, it randomly picks a country, then randomly selects an adjective, component, and specification from that country's defined characteristics.
# 3. Description Templates: It uses a few basic sentence structures to combine these elements.
# 4. Component-Specific Details: It then adds more specific, randomized details based on the chosen component type (e.g., resistance values for resistors, capacitance for capacitors) to make the descriptions richer and more realistic.
# 5. DataFrame Creation: Finally, it compiles all the generated descriptions and countries into a pandas DataFrame.

import pandas as pd
import random


def generate_synthetic_electronics_data(num_samples=500):
    """
    Generates a synthetic dataset of electronics part descriptions and their
    simulated countries of origin.

    Args:
        num_samples (int): The number of data samples to generate.

    Returns:
        pandas.DataFrame: A DataFrame with 'Part_Description' and 'Country_Of_Origin' columns.
    """

    # Define possible countries of origin and their associated typical characteristics/products
    countries = {
        "China": {
            "adjectives": [
                "cost-effective",
                "high-volume",
                "standard",
                "reliable",
                "mass-produced",
            ],
            "components": [
                "resistor",
                "capacitor",
                "LED",
                "diode",
                "transistor",
                "connector",
                "PCB",
                "power supply",
                "relay",
            ],
            "specs": [
                "SMD",
                "through-hole",
                "general purpose",
                "consumer grade",
                "industrial standard",
            ],
        },
        "USA": {
            "adjectives": [
                "high-performance",
                "military-grade",
                "aerospace",
                "specialized",
                "innovative",
                "rugged",
            ],
            "components": [
                "microcontroller",
                "FPGA",
                "ASIC",
                "sensor",
                "analog IC",
                "power management IC",
                "RF module",
            ],
            "specs": [
                "low-power",
                "high-speed",
                "customizable",
                "automotive-grade",
                "space-grade",
            ],
        },
        "Japan": {
            "adjectives": [
                "high-precision",
                "miniature",
                "automotive-grade",
                "ultra-low power",
                "reliable",
                "cutting-edge",
            ],
            "components": [
                "capacitor (ceramic/tantalum)",
                "inductor",
                "sensor (hall effect, pressure)",
                "crystal oscillator",
                "LCD driver",
                "camera module",
            ],
            "specs": ["compact", "high-frequency", "low-ESR", "long-life", "optical"],
        },
        "South Korea": {
            "adjectives": [
                "advanced",
                "high-density",
                "consumer electronics",
                "next-gen",
                "high-bandwidth",
            ],
            "components": [
                "memory IC (DRAM, NAND)",
                "display driver IC",
                "OLED panel",
                "processor",
                "power semiconductor",
            ],
            "specs": [
                "mobile-optimized",
                "high-resolution",
                "fast-charging",
                "compact design",
            ],
        },
        "Germany": {
            "adjectives": [
                "industrial-grade",
                "robust",
                "precision-engineered",
                "high-quality",
                "automotive-certified",
            ],
            "components": [
                "sensor (industrial)",
                "connector (heavy duty)",
                "relay (industrial)",
                "power module",
                "test equipment components",
            ],
            "specs": [
                "harsh environment",
                "high voltage",
                "certified",
                "safety-critical",
            ],
        },
        "Taiwan": {
            "adjectives": [
                "leading-edge",
                "foundry-produced",
                "chipset",
                "integrated",
                "innovative",
            ],
            "components": [
                "microcontroller",
                "logic IC",
                "power management IC",
                "network IC",
                "display panel components",
                "passive components",
            ],
            "specs": [
                "IoT-ready",
                "AI-enabled",
                "high-integration",
                "wireless",
                "compact",
            ],
        },
        "Vietnam": {
            "adjectives": [
                "assembly",
                "cost-efficient",
                "growing production",
                "consumer electronics",
            ],
            "components": [
                "cable assembly",
                "wiring harness",
                "simple PCB assembly",
                "LED light modules",
                "power adapters",
            ],
            "specs": ["manual assembly", "sub-assembly", "volume production"],
        },
        "Malaysia": {
            "adjectives": [
                "OSAT (assembly & test)",
                "discrete component",
                "semiconductor packaging",
                "high-volume",
            ],
            "components": [
                "transistor",
                "diode",
                "op-amp",
                "standard logic IC",
                "sensor assembly",
            ],
            "specs": ["reliable packaging", "component-level", "automotive assembly"],
        },
    }

    data = []
    country_list = list(countries.keys())

    for _ in range(num_samples):
        # Randomly choose a country for this sample
        country_of_origin = random.choice(country_list)
        country_info = countries[country_of_origin]

        # Select a random component and adjective/spec from the country's typical characteristics
        component = random.choice(country_info["components"])
        adjective = random.choice(country_info["adjectives"])
        spec = random.choice(country_info["specs"])

        # Create a basic description template
        description_templates = [
            f"{adjective} {component}, {spec}.",
            f"A {spec} {component} for {adjective} applications.",
            f"{component} ({adjective}) with {spec} features.",
            f"Manufactured for {adjective} needs: {component} with {spec}.",
        ]

        # Add more specific details based on component type
        part_details = ""
        if "resistor" in component:
            part_details = f" {random.choice(['10k ohm', '1k ohm', '220 ohm'])} {random.choice(['0.1%', '1%', '5%'])} tolerance, {random.choice(['0402', '0603', '1206'])}."
        elif "capacitor" in component:
            part_details = f" {random.choice(['100nF', '1uF', '10uF'])} {random.choice(['50V', '25V', '10V'])}, {random.choice(['ceramic', 'tantalum', 'electrolytic'])}."
        elif "microcontroller" in component:
            part_details = f" {random.choice(['ARM Cortex-M0', 'ESP32', 'ATMEGA328P'])} with {random.choice(['integrated Wi-Fi', 'low-power modes', 'high-speed ADC'])}."
        elif "LED" in component:
            part_details = f" {random.choice(['red', 'green', 'blue', 'white'])} {random.choice(['0603', '0805', 'SMD'])}."
        elif "connector" in component:
            part_details = f" {random.choice(['USB Type-C', 'HDMI', 'board-to-board'])} {random.choice(['20-pin', '4-pin', 'high-density'])}."
        elif "sensor" in component:
            part_details = f" {random.choice(['temperature', 'pressure', 'accelerometer'])} {random.choice(['digital output', 'analog output', 'I2C interface'])}."
        elif (
            "IC" in component
        ):  # For Integrated Circuits (analog, logic, power management)
            part_details = f" {random.choice(['op-amp', 'voltage regulator', 'logic gate'])} {random.choice(['SOIC-8', 'QFN-16', 'DIP-14'])}."
        elif "transistor" in component:
            part_details = f" {random.choice(['NPN BJT', 'MOSFET', 'IGBT'])} {random.choice(['TO-220', 'SOT-23'])}."
        elif "inductor" in component:
            part_details = f" {random.choice(['10uH', '100uH', '1mH'])} {random.choice(['SMD power inductor', 'ferrite core'])}."

        part_description = random.choice(description_templates) + part_details.strip()
        data.append(
            {
                "Part_Description": part_description,
                "Country_Of_Origin": country_of_origin,
            }
        )

    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":

    # Generate a dataset with n samples
    synthetic_df = generate_synthetic_electronics_data(num_samples=1000)
    print(synthetic_df.head())
    print("\nDataset Info:")
    print(synthetic_df.info())
    print("\nCountry Distribution:")
    print(synthetic_df["Country_Of_Origin"].value_counts())

    synthetic_df.to_csv("../data/synthetic_electronics_parts_1k.csv", index=False)
