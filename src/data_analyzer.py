"""
Statistical analysis and data processing - FIXED VERSION
"""

import pandas as pd
import numpy as np

class DataAnalyzer:
    def __init__(self, crop_df: pd.DataFrame, rainfall_df: pd.DataFrame, state_subdivision_map: dict):
        self.crop_df = crop_df
        self.rainfall_df = rainfall_df
        self.state_subdivision_map = state_subdivision_map
    
    def get_rainfall_data(self, state: str, years=None):
        """Get rainfall data for a state"""
        subdivisions = self.state_subdivision_map.get(state, [])
        if not subdivisions:
            return pd.DataFrame()
        
        df = self.rainfall_df[self.rainfall_df['SUBDIVISION'].isin(subdivisions)].copy()
        
        if years:
            if isinstance(years, str) and years.startswith("last_"):
                n = int(years.split("_")[1])
                max_year = df['YEAR'].max()
                years = list(range(max_year - n + 1, max_year + 1))
            
            if isinstance(years, list) and len(years) > 0:
                df = df[df['YEAR'].isin(years)]
        
        return df
    
    def get_crop_data(self, state=None, crop=None, years=None, district=None):
        """Get filtered crop production data"""
        df = self.crop_df.copy()
        
        if state:
            df = df[df['state_name'].str.lower() == state.lower()]
        
        if crop:
            # Try exact match first
            crop_match = df[df['crop'].str.lower() == crop.lower()]
            if not crop_match.empty:
                df = crop_match
            else:
                # Try partial match (e.g., "rice" matches "Rice", "Paddy")
                crop_match = df[df['crop'].str.contains(crop, case=False, na=False)]
                if not crop_match.empty:
                    df = crop_match
                    print(f"Matched '{crop}' to crops: {df['crop'].unique()[:3].tolist()}")
                else:
                    # Return empty dataframe with same columns
                    return pd.DataFrame(columns=self.crop_df.columns)
        
        if district:
            df = df[df['district_name'].str.lower() == district.lower()]
        
        # Check if df is empty before accessing crop_year
        if years and not df.empty:
            if isinstance(years, str) and years.startswith("last_"):
                n = int(years.split("_")[1])
                max_year = df['crop_year'].max()
                years = list(range(max_year - n + 1, max_year + 1))
            
            if isinstance(years, list) and len(years) > 0:
                df = df[df['crop_year'].isin(years)]
        
        return df
    
    def compare_rainfall_and_crops(self, state1: str, state2: str, years=None, top_n: int = 5):
        """Compare rainfall and top crops between two states"""
        # Get rainfall data
        rain1 = self.get_rainfall_data(state1, years)
        rain2 = self.get_rainfall_data(state2, years)
        
        if rain1.empty or rain2.empty:
            return None
        
        avg1 = rain1['ANNUAL'].mean()
        avg2 = rain2['ANNUAL'].mean()
        std1 = rain1['ANNUAL'].std()
        std2 = rain2['ANNUAL'].std()
        
        # Get top crops
        crop1 = self.get_crop_data(state1, years=years)
        crop2 = self.get_crop_data(state2, years=years)
        
        top_crops1 = (crop1.groupby('crop')['production_']
                     .sum()
                     .sort_values(ascending=False)
                     .head(top_n)
                     .reset_index())
        
        top_crops2 = (crop2.groupby('crop')['production_']
                     .sum()
                     .sort_values(ascending=False)
                     .head(top_n)
                     .reset_index())
        
        # Get year range
        if years and isinstance(years, list) and len(years) > 0:
            year_range = f"{min(years)}-{max(years)}"
        elif years and isinstance(years, str):
            year_range = years.replace("_", " ")
        else:
            year_range = "all available years"
        
        return {
            "rainfall": {
                state1: {
                    "mean": round(avg1, 2),
                    "std": round(std1, 2),
                    "subdivisions": self.state_subdivision_map.get(state1, [])
                },
                state2: {
                    "mean": round(avg2, 2),
                    "std": round(std2, 2),
                    "subdivisions": self.state_subdivision_map.get(state2, [])
                },
                "difference": round(abs(avg1 - avg2), 2),
                "higher": state1 if avg1 > avg2 else state2
            },
            "crops": {
                state1: top_crops1.to_dict('records'),
                state2: top_crops2.to_dict('records')
            },
            "year_range": year_range,
            "records_analyzed": {
                "rainfall": len(rain1) + len(rain2),
                "crops": len(crop1) + len(crop2)
            }
        }
    
    def compare_district_production(self, state1, state2, crop):
        """Compare district-level production between two states - FIXED"""
        
        df = self.crop_df.copy()
        
        # Normalize
        df["state_name"] = df["state_name"].str.strip().str.title()
        df["district_name"] = df["district_name"].str.strip().str.title()
        df["crop"] = df["crop"].str.strip().str.title()
        
        # Filter by crop
        crop_df = df[df["crop"].str.contains(crop, case=False, na=False)]
        
        if crop_df.empty:
            return None
        
        # Filter by states
        df_filtered = crop_df[crop_df["state_name"].isin([state1.title(), state2.title()])]
        
        if df_filtered.empty:
            return None
        
        # Aggregate by district and state
        summary = (
            df_filtered.groupby(["state_name", "district_name"], as_index=False)["production_"]
            .sum()
            .sort_values(by="production_", ascending=False)
        )
        
        # Get highest from each state
        state1_df = summary[summary["state_name"] == state1.title()]
        state2_df = summary[summary["state_name"] == state2.title()]
        
        if state1_df.empty or state2_df.empty:
            return None
        
        highest_state1 = state1_df.iloc[0]
        highest_state2 = state2_df.iloc[0]
        
        latest_year = int(df_filtered["crop_year"].max())
        
        # Return correct structure matching frontend expectations
        return {
            "highest": {
                "state": state1.title(),
                "district": highest_state1["district_name"],
                "production": float(highest_state1["production_"]),
                "year": latest_year,
                "crop": crop.title()
            },
            "second": {  # Changed from "lowest" to "second"
                "state": state2.title(),
                "district": highest_state2["district_name"],
                "production": float(highest_state2["production_"]),
                "year": latest_year,
                "crop": crop.title()
            },
            "ratio": round(highest_state1["production_"] / max(highest_state2["production_"], 1), 2),
            "crop": crop.title(),  # Add crop at top level
            "records_analyzed": len(df_filtered)
        }


    def analyze_trend_correlation(self, crop: str, state=None, years=None):
        """Analyze production trend and correlate with rainfall"""
        # Default to last 10 years if not specified
        if years is None:
            years = "last_10_years"
        
        crop_data = self.get_crop_data(state, crop, years)
        
        if crop_data.empty:
            return None
        
        # Production trend
        prod_trend = (crop_data.groupby('crop_year')['production_']
                     .sum()
                     .sort_index()
                     .reset_index())
        
        result = {
            "crop": crop,
            "state": state or "All India",
            "production_trend": prod_trend.to_dict('records'),
            "records_analyzed": len(crop_data)
        }
        
        # Add rainfall correlation if state specified
        if state:
            rain_data = self.get_rainfall_data(state, years)
            
            if not rain_data.empty:
                rain_trend = (rain_data.groupby('YEAR')['ANNUAL']
                             .mean()
                             .reset_index())
                
                # Merge and calculate correlation
                merged = pd.merge(
                    prod_trend, 
                    rain_trend,
                    left_on='crop_year',
                    right_on='YEAR',
                    how='inner'
                )
                
                if len(merged) >= 3:
                    correlation = np.corrcoef(merged['production_'], merged['ANNUAL'])[0, 1]
                    
                    # Calculate trend direction
                    prod_slope = np.polyfit(range(len(prod_trend)), prod_trend['production_'], 1)[0]
                    rain_slope = np.polyfit(range(len(rain_trend)), rain_trend['ANNUAL'], 1)[0]
                    
                    result['correlation'] = {
                        "coefficient": round(correlation, 3),
                        "interpretation": self._interpret_correlation(correlation),
                        "production_trend_direction": "increasing" if prod_slope > 0 else "decreasing",
                        "rainfall_trend_direction": "increasing" if rain_slope > 0 else "decreasing"
                    }
                    result['rainfall_trend'] = rain_trend.to_dict('records')
                    result['records_analyzed'] += len(rain_data)
        
        # Calculate year range - FIX HERE
        actual_years = crop_data['crop_year'].unique().tolist()
        if len(actual_years) > 0:
            result['years'] = f"{min(actual_years)}-{max(actual_years)}"
        elif isinstance(years, str):
            result['years'] = years.replace("_", " ")
        else:
            result['years'] = "unknown"
        
        return result
    
    def _interpret_correlation(self, corr: float) -> str:
        """Interpret correlation coefficient"""
        abs_corr = abs(corr)
        direction = "positive" if corr > 0 else "negative"
        
        if abs_corr < 0.3:
            strength = "Weak"
            impact = "minimal"
        elif abs_corr < 0.6:
            strength = "Moderate"
            impact = "partial"
        else:
            strength = "Strong"
            impact = "significant"
        
        return f"{strength} {direction} correlation - rainfall has {impact} impact on production"
    
    def policy_recommendation(self, crop_a: str, crop_b: str, state=None, years=None):
        """Generate data-backed policy recommendations - FIXED VERSION"""
        # Default to last 10 years if not specified
        if years is None or years == []:
            years = "last_10_years"
        
        data_a = self.get_crop_data(state, crop_a, years)
        data_b = self.get_crop_data(state, crop_b, years)
        
        if data_a.empty or data_b.empty:
            return {
                "error": f"Insufficient data for comparison. Found {len(data_a)} records for {crop_a} and {len(data_b)} records for {crop_b}."
            }
        
        arguments = []
        
        # Argument 1: Production Stability
        prod_a_yearly = data_a.groupby('crop_year')['production_'].sum()
        prod_b_yearly = data_b.groupby('crop_year')['production_'].sum()
        
        cv_a = prod_a_yearly.std() / prod_a_yearly.mean() if prod_a_yearly.mean() > 0 else 0
        cv_b = prod_b_yearly.std() / prod_b_yearly.mean() if prod_b_yearly.mean() > 0 else 0
        
        more_stable = crop_a if cv_a < cv_b else crop_b
        
        arguments.append({
            "argument": f"{more_stable} shows more stable production with lower year-to-year variation",
            "evidence": {
                f"{crop_a}_cv": round(cv_a, 3),
                f"{crop_b}_cv": round(cv_b, 3)
            },
            "interpretation": f"Coefficient of Variation: Lower is better. {more_stable} has {round(abs(cv_a - cv_b) * 100, 1)}% lower volatility."
        })
        
        # Argument 2: Total Production Volume
        total_a = data_a['production_'].sum()
        total_b = data_b['production_'].sum()
        
        higher_prod = crop_a if total_a > total_b else crop_b
        
        arguments.append({
            "argument": f"{higher_prod} demonstrates higher overall production capacity",
            "evidence": {
                f"{crop_a}_total": round(total_a, 0),
                f"{crop_b}_total": round(total_b, 0)
            },
            "interpretation": f"{higher_prod} produces {round(abs(total_a - total_b), 0)} tonnes more over the analyzed period."
        })
        
        # Argument 3: Land Efficiency
        if 'area_' in data_a.columns and data_a['area_'].sum() > 0 and data_b['area_'].sum() > 0:
            eff_a = total_a / data_a['area_'].sum()
            eff_b = total_b / data_b['area_'].sum()
            
            more_efficient = crop_a if eff_a > eff_b else crop_b
            
            arguments.append({
                "argument": f"{more_efficient} is more land-efficient with higher yield per hectare",
                "evidence": {
                    f"{crop_a}_efficiency": round(eff_a, 2),
                    f"{crop_b}_efficiency": round(eff_b, 2)
                },
                "interpretation": f"Production per hectare: {more_efficient} produces {round(abs(eff_a - eff_b), 2)} more tonnes/hectare."
            })
        else:
            # Argument 3 fallback: Growth trend
            if len(prod_a_yearly) >= 3 and len(prod_b_yearly) >= 3:
                growth_a = (prod_a_yearly.iloc[-1] - prod_a_yearly.iloc[0]) / prod_a_yearly.iloc[0] * 100
                growth_b = (prod_b_yearly.iloc[-1] - prod_b_yearly.iloc[0]) / prod_b_yearly.iloc[0] * 100
                
                growing_crop = crop_a if growth_a > growth_b else crop_b
                
                arguments.append({
                    "argument": f"{growing_crop} shows stronger growth trajectory over the period",
                    "evidence": {
                        f"{crop_a}_growth": f"{round(growth_a, 1)}%",
                        f"{crop_b}_growth": f"{round(growth_b, 1)}%"
                    },
                    "interpretation": f"Period-over-period growth: {growing_crop} increased by {max(abs(growth_a), abs(growth_b)):.1f}%."
                })
        
        # Calculate year range from actual data
        actual_years_a = data_a['crop_year'].unique().tolist()
        actual_years_b = data_b['crop_year'].unique().tolist()
        all_years = sorted(set(actual_years_a + actual_years_b))
        
        if len(all_years) > 0:
            year_range = f"{min(all_years)}-{max(all_years)}"
        elif isinstance(years, str):
            year_range = years.replace("_", " ")
        else:
            year_range = "unknown period"
        
        return {
            "recommendation": f"Promote {crop_a} over {crop_b}",
            "region": state or "All India",
            "time_period": year_range,
            "arguments": arguments,
            "records_analyzed": len(data_a) + len(data_b)
        }
    
    def get_top_crops(self, state=None, years=None, top_n: int = 10):
        """Get top crops by production"""
        data = self.get_crop_data(state, years=years)
        
        if data.empty:
            return None
        
        result = (data.groupby('crop')['production_']
                 .sum()
                 .sort_values(ascending=False)
                 .head(top_n)
                 .reset_index())
        
        return {
            "crops": result.to_dict('records'),
            "state": state or "All India",
            "records_analyzed": len(data)
        }