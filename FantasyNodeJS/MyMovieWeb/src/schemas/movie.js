var mongoose = require("mongoose");

var MoviewSchema = new mongoose.Schema({
    derector:String,
    title:String,
    language:String,
    country:String,
    summary:String,
    flash:String,
    poster:String,
    year:Number,
    meta:{
        
    }
});