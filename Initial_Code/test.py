from FA_SIM.Single_agent_case.Initial_Code.utils import (
    generate_ap_locations, 
    generate_ue_locations,
    generate_path_angles,
    generate_path_response_matrix
) 

if __name__ == "__main__":
    positions = generate_ap_locations(
        ap_nums=6,
        area_size=200.0
    )

    print(positions)
    print("Shape:", positions.shape)