import asyncio
import pandas as pd
import logging

from tqdm import tqdm
logging.getLogger("prophet.plot").disabled = True
logging.getLogger("cmdstanpy").disabled=True
from prophet import Prophet
from sf_platform import ServerlessPlatform
from heuristic import ServerlessScheduler
from datetime import datetime
from utils import load_azure_data, preprocess_azure_data


def new_prophet():
    prophet = Prophet(changepoint_prior_scale=1)
    prophet.add_seasonality(name='seconds_level', period=60, fourier_order=20)
    prophet.add_seasonality(name="minute_level", period=3600, fourier_order=5)
    return prophet

async def process_trace_dataframe(sch: ServerlessScheduler, df: pd.DataFrame, function_name: str):
    df_instances = pd.DataFrame({"ds": [], "num": []})
    for idx, row in tqdm(df.iterrows()):
        target_time = row["ds"]
        value = row["y"]

        now = datetime.now()
        if target_time > now:
            await asyncio.sleep((target_time - now).total_seconds())
        
        for _ in range(int(value)):
            # print("RUNNING FUNCTION")
            asyncio.create_task(sch.platform.run_function(function_name, query_params="t=0.3"))
        
        # Add number of available instances at each time
        df_instances.loc[len(df_instances)] = {"ds": target_time, "num": len(sch.platform._available_instances[function_name])}

        if (idx + 1) % 10 == 0: # idk whatever lmao
            new_model = new_prophet()
            new_model.fit(df.iloc[:idx])
            sch.models[function_name] = new_model
            asyncio.create_task(sch.adjust_scheduling())

    return df_instances

async def main(samples=100, warming_period=30, use_model=True):
    sp = ServerlessPlatform(time_multiplier=10)
    await sp.register_function("sleep", "./sf_platform/examples/sleep/entry.py", "./sf_platform/examples/sleep/requirements.txt")

    scheduler = ServerlessScheduler(sp, { "sleep": new_prophet() }, gen_pred=use_model, default_warm_period=warming_period)

    # uncomment if you don't have the data
    # load_azure_data()
    trace = preprocess_azure_data(num_samples=samples, bin_size=1)
    print(trace.head())
    instances = await process_trace_dataframe(scheduler, trace, "sleep")
    
    sp.shutdown()

    return trace, scheduler, instances

if __name__ == "__main__":
    asyncio.run(main())
