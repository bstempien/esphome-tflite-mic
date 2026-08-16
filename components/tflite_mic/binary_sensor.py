import esphome.codegen as cg
import esphome.config_validation as cv
from esphome.components import binary_sensor
from esphome.const import CONF_ID

from . import tflite_mic_ns, TFLiteMicComponent

DEPENDENCIES = ["tflite_mic"]

CONF_TFLITE_MIC_ID = "tflite_mic_id"
CONF_CLASS_INDEX = "class_index"
CONF_THRESHOLD = "threshold"
CONF_MIN_TRIGGER_INTERVAL = "min_trigger_interval"

TFLiteMicBinarySensor = tflite_mic_ns.class_(
    "TFLiteMicBinarySensor", binary_sensor.BinarySensor, cg.Component
)

CONFIG_SCHEMA = binary_sensor.binary_sensor_schema(TFLiteMicBinarySensor).extend(
    {
        cv.GenerateID(CONF_TFLITE_MIC_ID): cv.use_id(TFLiteMicComponent),
        cv.Required(CONF_CLASS_INDEX): cv.int_range(min=0, max=63),
        cv.Optional(CONF_THRESHOLD, default=0.8): cv.percentage,
        cv.Optional(CONF_MIN_TRIGGER_INTERVAL, default="1000ms"): cv.positive_time_period_milliseconds,
    }
).extend(cv.COMPONENT_SCHEMA)


async def to_code(config):
    var = await binary_sensor.new_binary_sensor(config)
    await cg.register_component(var, config)

    parent = await cg.get_variable(config[CONF_TFLITE_MIC_ID])
    cg.add(var.set_class_index(config[CONF_CLASS_INDEX]))
    cg.add(var.set_threshold(config[CONF_THRESHOLD]))
    cg.add(var.set_min_trigger_interval(config[CONF_MIN_TRIGGER_INTERVAL].total_milliseconds))
    cg.add(parent.register_sensor(var))
