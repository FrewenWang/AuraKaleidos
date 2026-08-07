declare module "connect-mongo" {
    import session from "express-session";

    interface MongoStoreOptions {
        url: string;
        autoReconnect?: boolean;
    }

    type MongoStoreConstructor = new (options: MongoStoreOptions) => session.Store;

    function connectMongo(sessionModule: typeof session): MongoStoreConstructor;

    export = connectMongo;
}
