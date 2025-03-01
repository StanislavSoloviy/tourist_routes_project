import { GET_LOCAL_SPOTS, IS_SPOTS_LOADING, SPOTS_ERROR } from "./spotsActionTypeNames";
import { TSpotsActions } from "modules/card-list/store/types/spotsActionTypes";
import { TSpotsState } from "./types/spotsReducerTypes";

const initialState: TSpotsState = {
    data: null,
    total: null,
    loading: false,
    error: false
};

export function spotsReducer(state = initialState, action:TSpotsActions) {

    switch (action.type) {
        case GET_LOCAL_SPOTS: 
            return {...state, data: action.payload, total: action.payload.length}         
        case SPOTS_ERROR:
            return {...state, error: action.payload}
        case IS_SPOTS_LOADING:
            return {...state, loading: action.payload}
        default:
            return state
    }
}