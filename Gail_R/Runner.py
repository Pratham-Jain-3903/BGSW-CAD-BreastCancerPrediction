# import pymysql
# import pandas as pd

# # Connection settings (ensure to replace with your actual connection details)
# db_connection = pymysql.connect(
#     host='localhost',  # replace with your DB host
#     user='your_username',  # replace with your DB username
#     password='your_password',  # replace with your DB password
#     database='breastriskcalc'  # replace with your DB name
# )

# # SQL query to select all white women
# sql_query = """
#     SELECT DISTINCT mrn
#     FROM athena_survey
#     WHERE white = 1
# """
# data = pd.read_sql(sql_query, db_connection)

# # Retrieve all entries for these patients (g_5yearscore or vt_gailrisk*100)
# mrn_list = ",".join(map(str, data['mrn'].tolist()))
# sql_query = f"""
#     SELECT mrn, vt_gailrisk*100 AS vt_gailrisk
#     FROM athena_survey
#     WHERE mrn IN ({mrn_list})
#     ORDER BY mrn, visitdate
# """
# data = pd.read_sql(sql_query, db_connection)

# # Create empty dataframe for results
# df = pd.DataFrame(columns=["MRN", "INIT", "LAST"])

# # Initialize variables
# mrn_prev = None
# g_s_init = 0
# g_s_last = 100
# exams = 0
# df_idx = 0

# # Loop through the data to process each patient
# for i, row in data.iterrows():
#     mrn = row['mrn']
#     g_s = row['vt_gailrisk']
    
#     if pd.isna(g_s):
#         continue  # Skip rows without Gail scores

#     # Check if we have a new patient
#     if mrn_prev != mrn:
#         if g_s_last <= g_s_init and exams > 1:
#             # Append the patient to the dataframe
#             df.loc[df_idx] = [mrn_prev, g_s_init, g_s_last]
#             df_idx += 1
        
#         # Reset initial Gail score and exam count
#         g_s_init = g_s
#         exams = 0

#     # Update previous MRN and scores
#     mrn_prev = mrn
#     g_s_last = g_s
#     exams += 1

# # Remove unpopulated rows
# df = df.dropna()

# # Write results to file
# output_file = "/Users/simonxhan/repos/breastriskcalc/decGailPatientsList.txt"
# df['MRN'].to_csv(output_file, index=False, header=False, sep=',', line_terminator=',\n')

# # Close the database connection
# db_connection.close()


import sqlite3
import pandas as pd
import numpy as np

def get_patient_data():
    """
    Fetches distinct MRNs for white women and their Gail scores from the SQLite database.
    Returns the fetched data as a pandas DataFrame.
    """
    # Connect to the SQLite database
    db_connection = sqlite3.connect('breastriskcalc.db')  # Update with the path to your SQLite database

    # Fetch distinct MRNs for white women
    sql_query = """
        SELECT DISTINCT mrn
        FROM athena_survey
        WHERE white = 1
    """
    data = pd.read_sql(sql_query, db_connection)

    # Retrieve Gail risk scores for those patients
    mrn_list = ",".join(map(str, data['mrn'].tolist()))
    sql_query = f"""
        SELECT mrn, vt_gailrisk*100 AS vt_gailrisk
        FROM athena_survey
        WHERE mrn IN ({mrn_list})
        ORDER BY mrn, visitdate
    """
    data = pd.read_sql(sql_query, db_connection)

    # Close the connection
    db_connection.close()

    return data

def process_patient_data(data):
    """
    Processes the fetched patient data to find those with decreasing 5-year Gail scores.
    Returns a DataFrame with MRN, initial score, and last score for such patients.
    """
    df = pd.DataFrame(columns=["MRN", "INIT", "LAST"])
    mrn_prev = None
    g_s_init = 0
    g_s_last = 100
    exams = 0
    df_idx = 0

    for i, row in data.iterrows():
        mrn = row['mrn']
        g_s = row['vt_gailrisk']

        if pd.isna(g_s):
            continue  # Skip rows without Gail scores

        if mrn_prev != mrn:  # New patient
            if g_s_last <= g_s_init and exams > 1:
                df.loc[df_idx] = [mrn_prev, g_s_init, g_s_last]
                df_idx += 1
            g_s_init = g_s
            exams = 0

        mrn_prev = mrn
        g_s_last = g_s
        exams += 1

    df = df.dropna()
    return df

def write_output(df):
    """
    Writes the processed patient data to a file.
    """
    output_file = r"C:\Users\Pratham Jain\SisterDear\AIHC\Gail\decGailPatientsList.txt"
    df['MRN'].to_csv(output_file, index=False, header=False, sep=',', line_terminator=',\n')
    print(f"Output written to {output_file}")

def run_risk_calculator():
    """
    Example of setting up risk calculation using matrices.
    """

    # Competing hazards - non-breast cancer hazards (h2)
    h2_ = np.array([
        [0.000493, 0.00074354, 0.000437, 0.0004412, 0.00074354, 0.000437, 0.000210649076, 0.000173593803, 0.000229120979, 0.000563507269, 0.000465500812, 0.000212632332], # 0. [20:25)
    [0.000531, 0.00101698, 0.000533, 0.0005254, 0.00101698, 0.000533, 0.000192644865, 0.000295805882, 0.000262988494, 0.000369640217, 0.000600466920, 0.000242170741], # 1. [25:30)
    [0.000625, 0.00145937, 0.000700, 0.0006746, 0.00145937, 0.000700, 0.000244435215, 0.000228322534, 0.000314844090, 0.001019912579, 0.000851057138, 0.000301552711], # 2. [30:35)
    [0.000825, 0.00215933, 0.000897, 0.0009092, 0.00215933, 0.000897, 0.000317895949, 0.000363242389, 0.000394471908, 0.001234013911, 0.001478265376, 0.000369053354], # 3. [35:40)
    [0.001307, 0.00315077, 0.001163, 0.0012534, 0.00315077, 0.001163, 0.000473261994, 0.000590633044, 0.000647622610, 0.002098344078, 0.001931486788, 0.000543002943], # 4. [40:45)
    [0.002181, 0.00448779, 0.001720, 0.0019570, 0.00448779, 0.001720, 0.000800271380, 0.001086079485, 0.001170202327, 0.002982934175, 0.003866623959, 0.000893862331], # 5. [45:50)
    [0.003655, 0.00632281, 0.002646, 0.0032984, 0.00632281, 0.002646, 0.001217480226, 0.001859999966, 0.001809380379, 0.005402445702, 0.004924932309, 0.001515172239], # 6. [50:55)
    [0.005852, 0.00963037, 0.004216, 0.0054622, 0.00963037, 0.004216, 0.002099836508, 0.003216600974, 0.002614170568, 0.009591474245, 0.008177071806, 0.002574669551], # 7. [55:60)
    [0.009439, 0.01471818, 0.006960, 0.0091035, 0.01471818, 0.006960, 0.003436889186, 0.004719402141, 0.004483330681, 0.016315472607, 0.008638202890, 0.004324370426], # 8. [60:65)
    [0.015028, 0.02116304, 0.010867, 0.0141854, 0.02116304, 0.010867, 0.006097405623, 0.008535331402, 0.007393665092, 0.020152229069, 0.018974658371, 0.007419621918], # 9. [65:70)
    [0.023839, 0.03266035, 0.016858, 0.0225935, 0.03266035, 0.016858, 0.010664526765, 0.012433511681, 0.012233059675, 0.027354838710, 0.029257567105, 0.013251765130], # 10. [70:75)
    [0.038832, 0.04564087, 0.025156, 0.0361146, 0.04564087, 0.025156, 0.020148678452, 0.020230197885, 0.021127058106, 0.050446998723, 0.038408980974, 0.022291427490], # 11. [75:80)
    [0.066828, 0.06835185, 0.041866, 0.0613626, 0.06835185, 0.041866, 0.037990796590, 0.037725498348, 0.037936954809, 0.072262026612, 0.052869579345, 0.041746550635], # 12. [80:85)
    [0.144908, 0.13271262, 0.089476, 0.1420663, 0.13271262, 0.089476, 0.098333900733, 0.106149118663, 0.085138518334, 0.145844504021, 0.074745721133, 0.087485802065]  # 13. [85:90)
    ])

    # Breast cancer composite incidence rates (h1*)
    h1_star_ = np.array([
        [0.000010, 0.00002696, 0.000020, 0.0000122, 0.00002696, 0.000020, 0.000004059636, 0.000000000001, 0.000007500161, 0.000045080582, 0.000000000001, 0.000012355409], # 0. [20:25)
    [0.000076, 0.00011295, 0.000071, 0.0000741, 0.00011295, 0.000071, 0.000045944465, 0.000099483924, 0.000081073945, 0.000098570724, 0.000071525212, 0.000059526456], # 1. [25:30)
    [0.000266, 0.00031094, 0.000197, 0.0002297, 0.00031094, 0.000197, 0.000188279352, 0.000287041681, 0.000227492565, 0.000339970860, 0.000288799028, 0.000184320831], # 2. [30:35)
    [0.000661, 0.00067639, 0.000438, 0.0005649, 0.00067639, 0.000438, 0.000492930493, 0.000545285759, 0.000549786433, 0.000852591429, 0.000602250698, 0.000454677273], # 3. [35:40)
    [0.001265, 0.00119444, 0.000811, 0.0011645, 0.00119444, 0.000811, 0.000913603501, 0.001152211095, 0.001129400541, 0.001668562761, 0.000755579402, 0.000791265338], # 4. [40:45)
    [0.001866, 0.00187394, 0.001307, 0.0019525, 0.00187394, 0.001307, 0.001471537353, 0.001859245108, 0.001813873795, 0.002552703284, 0.000766406354, 0.001048462801], # 5. [45:50)
    [0.002211, 0.00241504, 0.001574, 0.0026154, 0.00241504, 0.001574, 0.001421275482, 0.002606291272, 0.002223665639, 0.003321774046, 0.001893124938, 0.001372467817], # 6. [50:55)
    [0.002721, 0.00291112, 0.001857, 0.0030279, 0.00291112, 0.001857, 0.001970946494, 0.003221751682, 0.002680309266, 0.005373001776, 0.002365580107, 0.001495473711], # 7. [55:60)
    [0.003348, 0.00310127, 0.002151, 0.0036757, 0.00310127, 0.002151, 0.001674745804, 0.004006961859, 0.002891219230, 0.005237808549, 0.002843933070, 0.001646746198], # 8. [60:65)
    [0.003923, 0.00366560, 0.002512, 0.0042029, 0.00366560, 0.002512, 0.001821581075, 0.003521715275, 0.002534421279, 0.005581732512, 0.002920921732, 0.001478363563], # 9. [65:70)
    [0.004178, 0.00393132, 0.002846, 0.0047308, 0.00393132, 0.002846, 0.001834477198, 0.003593038294, 0.002457159409, 0.005677419355, 0.002330395655, 0.001216010125], # 10. [70:75)
    [0.004439, 0.00408951, 0.002757, 0.0049425, 0.00408951, 0.002757, 0.001919911972, 0.003589303081, 0.002286616920, 0.006513409962, 0.002036291235, 0.001067663700], # 11. [75:80)
    [0.004421, 0.00396793, 0.002523, 0.0047976, 0.00396793, 0.002523, 0.002233371071, 0.003538507159, 0.001814802825, 0.003889457523, 0.001482683983, 0.001376104012], # 12. [80:85)
    [0.004109, 0.00363712, 0.002039, 0.0040106, 0.00363712, 0.002039, 0.002247315779, 0.002051572909, 0.001750879130, 0.002949061662, 0.001012248203, 0.000661576644]  # 13. [85:90)
    ])

    # Logistic regression coefficients (beta)
    beta_ = np.array([
        [-0.7494824600, -0.3457169653, -0.7494824600, -0.7494824600, -0.3457169653, -0.7494824600], # Intercept
    [0.0108080720,  0.0334703319,  0.0108080720,  0.0108080720,  0.0334703319,  0.0108080720], # Age >= 50 indicator
    [0.0940103059,  0.2672530336,  0.0940103059,  0.0940103059,  0.2672530336,  0.0940103059], # Age menarche
    [0.5292641686,  0.1822121131,  0.5292641686,  0.5292641686,  0.1822121131,  0.5292641686], # Num of breast biopsies
    [0.2186262218,  0.0000000000,  0.2186262218,  0.2186262218,  0.0000000000,  0.2186262218], # Age 1st live birth
    [0.9583027845,  0.4757242578,  0.9583027845,  0.9583027845,  0.4757242578,  0.9583027845], # 1st degree relatives
    [-0.2880424830, -0.1119411682, -0.2880424830, -0.2880424830, -0.1119411682, -0.2880424830], # Breast biopsies * age >= 50
    [-0.1908113865,  0.0000000000, -0.1908113865, -0.1908113865,  0.0000000000, -0.1908113865]  # Age 1st live birth * 1st degree relatives
    ])


    # Example logic to use the matrices for calculations
    # Implement your risk calculation logic here based on h2_, h1_star_, and beta_

    # For demonstration, just print the shapes of the arrays
    print(f"h2_ shape: {h2_.shape}")
    print(f"h1_star_ shape: {h1_star_.shape}")
    print(f"beta_ shape: {beta_.shape}")

def main():
    # Step 1: Fetch the data from the database
    print("Fetching patient data from the database...")
    data = get_patient_data()

    # Step 2: Process the data to find decreasing Gail scores
    print("Processing patient data for decreasing Gail scores...")
    df = process_patient_data(data)

    # Step 3: Write the processed data to a file
    print("Writing output to file...")
    write_output(df)

    # Step 4: (Optional) Run the risk calculator
    print("Running risk calculator...")
    run_risk_calculator()

if __name__ == "__main__":
    main()
