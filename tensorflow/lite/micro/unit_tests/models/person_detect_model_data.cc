#include <cstdint>

#include "person_detect_model_data.h"

unsigned int g_person_detect_model_data_size = 300568;


unsigned char *GetPersonDetectModelData() {
    return g_person_detect_model_data;
}

unsigned int GetPersonDetectModelDataSize() {
    return g_person_detect_model_data_size;
}